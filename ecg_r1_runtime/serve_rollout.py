"""Local Swift rollout server entry point for standalone inference.

Swift's upstream rollout command is designed for GRPO training and forces
``load_format='dummy'`` so weights can be synchronized later. This entry point
keeps the same HTTP API but loads checkpoint weights directly.
"""

from __future__ import annotations

import os
import asyncio
from typing import Optional, Union
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

from swift.llm import RolloutArguments
from swift.llm.infer.infer_engine import GRPOVllmEngine
from swift.llm.infer.rollout import SwiftRolloutDeploy, get_rollout_engine_type
from swift.utils import get_logger

logger = get_logger()


def standalone_llm_worker(args: RolloutArguments, data_parallel_rank: int, master_port: int,
                          connection: Connection) -> None:
    args._import_external_plugins()
    args._init_custom_register()
    os.environ["VLLM_DP_RANK"] = str(data_parallel_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(data_parallel_rank)
    os.environ["VLLM_DP_SIZE"] = str(args.vllm_data_parallel_size)
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    engine = StandaloneRolloutDeploy.get_infer_engine(args, template=args.get_template(None))
    rollout_engine = get_rollout_engine_type(args, engine)
    connection.send({"status": "ready"})

    while True:
        try:
            command = connection.recv()
        except KeyboardInterrupt:
            engine.engine.collective_rpc(method="close_communicator")
            break

        if command["type"] in ["call", "fire_and_forget"]:
            method_name = command["method"]
            args_, kwargs = command.get("args", ()), command.get("kwargs", {})
            method = getattr(rollout_engine, method_name, None) or getattr(rollout_engine.engine, method_name, None)
            result = method(*args_, **kwargs)
            if command["type"] == "call":
                connection.send(result)
        elif command["type"] == "shutdown":
            break


async def standalone_async_llm_worker(args: RolloutArguments, data_parallel_rank: int, master_port: int,
                                      connection: Connection) -> None:
    args._import_external_plugins()
    args._init_custom_register()
    engine = StandaloneRolloutDeploy.get_infer_engine(args, template=args.get_template(None))
    rollout_engine = get_rollout_engine_type(args, engine)
    connection.send({"status": "ready"})

    loop = asyncio.get_running_loop()
    while True:
        try:
            command = await loop.run_in_executor(None, connection.recv)
        except KeyboardInterrupt:
            await engine.engine.collective_rpc(method="close_communicator")
            break

        if command["type"] in ["call", "fire_and_forget"]:
            method_name = command["method"]
            args_, kwargs = command.get("args", ()), command.get("kwargs", {})
            method = getattr(rollout_engine, method_name, None) or getattr(rollout_engine.engine, method_name, None)
            result = await method(*args_, **kwargs)
            if command["type"] == "call":
                connection.send(result)
        elif command["type"] == "shutdown":
            break


def standalone_llm_worker_entry(*args, **kwargs):
    asyncio.run(standalone_async_llm_worker(*args, **kwargs))


class StandaloneRolloutDeploy(SwiftRolloutDeploy):
    def _start_data_parallel_workers(self):
        for data_parallel_rank in range(self.num_connections):
            parent_conn, child_conn = Pipe()
            worker_func = standalone_llm_worker_entry if self.use_async_engine else standalone_llm_worker
            process = Process(target=worker_func, args=(self.args, data_parallel_rank, self.master_port, child_conn))
            process.start()
            self.connections.append(parent_conn)
            self.processes.append(process)

    @staticmethod
    def get_infer_engine(args: RolloutArguments, template=None, **kwargs):
        kwargs.update({
            "model_id_or_path": args.model,
            "model_type": args.model_type,
            "revision": args.model_revision,
            "torch_dtype": args.torch_dtype,
            "template": template,
            "use_async_engine": args.vllm_use_async_engine,
        })
        infer_backend = kwargs.pop("infer_backend", None) or args.infer_backend
        if infer_backend != "vllm":
            infer_backend = "vllm"
            logger.info("Currently, rollout only supports the vLLM backend. Set vLLM backend")

        kwargs.update(args.get_vllm_engine_kwargs())
        engine_kwargs = kwargs.get("engine_kwargs", {})
        engine_kwargs.update({"worker_extension_cls": "trl.scripts.vllm_serve.WeightSyncWorkerExtension"})
        engine_kwargs["load_format"] = os.environ.get("VLLM_LOAD_FORMAT", "auto")

        if args.vllm_use_async_engine and args.vllm_data_parallel_size > 1:
            engine_kwargs["data_parallel_size"] = args.vllm_data_parallel_size
        kwargs["engine_kwargs"] = engine_kwargs

        return GRPOVllmEngine(**kwargs)


def rollout_main(args: Optional[Union[list[str], RolloutArguments]] = None) -> None:
    StandaloneRolloutDeploy(args).main()


if __name__ == "__main__":
    rollout_main()
