from gpu_sage.core.models import Job, JobStatus
from gpu_sage.evaluation.metrics import compute_metrics


def test_metrics_calculation() -> None:
    jobs = [
        Job(1, 0, 1, 16, 10, start_time=2, completion_time=12, status=JobStatus.COMPLETED),
        Job(2, 0, 2, 16, 20, start_time=5, completion_time=25, status=JobStatus.COMPLETED),
    ]
    metrics = compute_metrics(jobs, total_gpus=4, simulated_time=25, gpu_time_used=35)
    assert metrics.completed_jobs == 2
    assert metrics.average_waiting_time == 3.5
    assert metrics.throughput_jobs_per_time == 2 / 25
    assert metrics.gpu_utilization == 35 / 100
