from typing import List

from .schemas import KarakaRole, KarakaTaskNode, RoleProgression


def derive_active_roles(node: KarakaTaskNode) -> List[KarakaRole]:
    base_roles: List[KarakaRole] = [
        KarakaRole.KARTA,
        KarakaRole.KARMA,
        KarakaRole.KARANA,
    ]
    if node.role_phase == RoleProgression.INIT:
        return base_roles + [KarakaRole.ADHIKARANA]
    if node.role_phase == RoleProgression.TARGET:
        return base_roles + [KarakaRole.SAMPRADANA, KarakaRole.ADHIKARANA]
    if node.role_phase == RoleProgression.BOUNDARY:
        return base_roles + [
            KarakaRole.APADANA,
            KarakaRole.SAMPRADANA,
            KarakaRole.ADHIKARANA,
        ]
    return base_roles + [KarakaRole.ADHIKARANA]
