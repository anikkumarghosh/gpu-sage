from gpu_sage.core.cluster import Cluster
from gpu_sage.core.models import Job


def test_scheduler_action_is_one_job_at_a_time() -> None:
    cluster = Cluster.homogeneous(8, 80)
    jobs = [
        Job(1, 0.0, 4, 32, 100),
        Job(2, 0.0, 2, 16, 20),
    ]
    assert cluster.can_allocate(jobs[0])
    assert cluster.can_allocate(jobs[1])
    cluster.allocate(jobs[0])
    assert cluster.free_gpu_count == 4
    assert not cluster.can_allocate(Job(3, 0.0, 5, 16, 20))
