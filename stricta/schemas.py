from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class KarakaRole(str, Enum):
    KARTA = "Kartā"
    KARMA = "Karma"
    KARANA = "Karaṇa"
    SAMPRADANA = "Sampradāna"
    APADANA = "Apādāna"
    ADHIKARANA = "Adhikaraṇa"


class RoleProgression(str, Enum):
    INIT = "INIT"
    TARGET = "TARGET"
    BOUNDARY = "BOUNDARY"
    COMPLETE = "COMPLETE"


class NodeStatus(str, Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"


@dataclass
class KarakaTaskNode:
    node_id: str
    objective_description: str
    karta: str
    karma: str
    karana: str
    sampradana: str
    apadana: str
    adhikarana: str
    dependencies: List[str] = field(default_factory=list)
    role_phase: RoleProgression = RoleProgression.INIT
    status: NodeStatus = NodeStatus.PENDING
    retry_count: int = 0
    containment_tightness: float = 1.0


@dataclass
class ActionProposal:
    action_id: str
    target_node_id: str
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    expected_state_change: str = ""
    confidence_score: float = 0.0
    tightness_factor: float = 1.0


@dataclass
class CriticVerdict:
    accepted: bool
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepSnapshot:
    node_state: KarakaTaskNode
    runtime_state: Dict[str, Any]
    created_at: Optional[str] = None
