# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import socket
from typing import Optional

import torch
from nccl.core.communicator import Communicator
from nccl.core.utils import UniqueId, get_unique_id


def _listening_tcp_ports() -> set[int]:
    """Return TCP listen ports in the current network namespace."""
    ports: set[int] = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path) as table:
                next(table, None)
                for line in table:
                    fields = line.split()
                    if len(fields) >= 4 and fields[3] == "0A":
                        ports.add(int(fields[1].rsplit(":", 1)[1], 16))
        except (OSError, ValueError):
            continue
    return ports


class StatelessProcessGroup:
    def __init__(self, master_address: str, port: int, rank: int, world_size: int):
        self.master_address = master_address
        self.port = port
        self.rank = rank
        self.world_size = world_size
        self.tcp_store = torch.distributed.TCPStore(
            host_name=self.master_address,
            port=self.port,
            world_size=self.world_size,
            is_master=(self.rank == 0),
        )

    def init_nccl_communicator(self, device: int):
        UNIQUE_ID_KEY = "nccl_unique_id"
        context = (
            f"rank={self.rank}/{self.world_size} host={socket.gethostname()} "
            f"pid={os.getpid()} device={device} "
            f"socket_ifname={os.environ.get('NCCL_SOCKET_IFNAME')} "
            f"comm_id={os.environ.get('NCCL_COMM_ID')}"
        )
        print(f"[SPG_DIAG] event=init_start {context}", flush=True)

        if self.rank == 0:
            listeners_before = _listening_tcp_ports()
            unique_id = get_unique_id()
            root_listener_candidates = _listening_tcp_ports() - listeners_before
            unique_id_bytes = unique_id.as_bytes
            # Rank 0: store unique_id to TCPStore
            self.tcp_store.set(UNIQUE_ID_KEY, unique_id_bytes)
            listeners_still_open = _listening_tcp_ports() & root_listener_candidates
            print(
                f"[SPG_DIAG] event=unique_id_published {context} "
                f"new_listen_ports={sorted(root_listener_candidates)} "
                f"still_open={sorted(listeners_still_open)}",
                flush=True,
            )
        else:
            # Other ranks: get unique_id from TCPStore
            self.tcp_store.wait([UNIQUE_ID_KEY])
            unique_id_bytes = self.tcp_store.get(UNIQUE_ID_KEY)
            unique_id = UniqueId.from_bytes(unique_id_bytes)

        with torch.cuda.device(device):
            print(
                f"[SPG_DIAG] event=comm_init_start {context} "
                f"allocated={torch.cuda.memory_allocated(device)} "
                f"reserved={torch.cuda.memory_reserved(device)}",
                flush=True,
            )
            self.nccl_communicator = Communicator.init(
                nranks=self.world_size,
                rank=self.rank,
                unique_id=unique_id,
            )
            print(f"[SPG_DIAG] event=comm_init_complete {context}", flush=True)
            # warmup and check if broadcast is working
            stream = torch.cuda.current_stream()
            if self.rank == 0:
                data = torch.ones(1, device=device)
            else:
                data = torch.zeros(1, device=device)
            self.broadcast(data, 0, stream=stream)
            torch.cuda.current_stream().synchronize()
            assert torch.allclose(data, torch.ones(1, device=device))

    def broadcast(
        self, tensor: torch.Tensor, src: int, stream: Optional[torch.cuda.Stream] = None
    ):
        if stream is None:
            stream = torch.cuda.current_stream()
        self.nccl_communicator.broadcast(
            sendbuf=tensor, recvbuf=tensor, root=src, stream=int(stream.cuda_stream)
        )
