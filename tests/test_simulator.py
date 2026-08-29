from gpu_sage.core.cluster import Cluster
from gpu_sage.core.models import Job, JobStatus
from gpu_sage.core.simulator import Simulator
from gpu_sage.schedulers.fcfs import FCFSScheduler


def test_single_job_lifecycle() -> None:
    job = Job(
        job_id=1,
        arrival_time=5.0,
        gpu_count=2,
        gpu_memory_gb=16.0,
        duration=10.0,
    )
    sim = Simulator(Cluster.homogeneous(4), FCFSScheduler())
    sim.load_jobs([job])
    sim.run()

    assert job.status == JobStatus.COMPLETED
    assert job.start_time == 5.0
    assert job.completion_time == 15.0
    assert job.waiting_time == 0.0
    assert sim.cluster.free_gpu_count == 4


def test_fcfs_orders_feasible_jobs_by_arrival() -> None:
    jobs = [
        Job(1, 0.0, 4, 16, 20),
        Job(2, 1.0, 2, 16, 20),
        Job(3, 2.0, 1, 16, 20),
    ]
    sim = Simulator(Cluster.homogeneous(4), FCFSScheduler())
    sim.load_jobs(jobs)
    sim.run()

    assert jobs[0].start_time == 0.0
    assert jobs[1].start_time is not None
    assert jobs[2].start_time is not None
    assert jobs[1].start_time <= jobs[2].start_time


def test_incompatible_memory_job_is_not_started() -> None:
    job = Job(1, 0.0, 1, 100.0, 10.0)
    sim = Simulator(Cluster.homogeneous(2, memory_gb=80.0), FCFSScheduler())
    sim.load_jobs([job])
    sim.run()

    assert job.status == JobStatus.WAITING
    assert job.start_time is None
    assert sim.cluster.free_gpu_count == 2
