"""Run milestone-1 simulator demo."""

from gpu_sage.core.cluster import Cluster
from gpu_sage.core.simulator import Simulator
from gpu_sage.schedulers.fcfs import FCFSScheduler
from gpu_sage.workloads.generator import SyntheticWorkload, WorkloadConfig


def main() -> None:
    config = WorkloadConfig(
        arrival_rate=0.08,
        min_gpus=1,
        max_gpus=4,
        min_duration=25,
        max_duration=120,
    )
    jobs = SyntheticWorkload(config, seed=42).generate(20)

    simulator = Simulator(cluster=Cluster.homogeneous(8, memory_gb=80), scheduler=FCFSScheduler())
    simulator.load_jobs(jobs)
    history = simulator.run()

    print("GPU-Sage milestone 1")
    print(f"Simulated time: {simulator.current_time:.2f}s")
    print(f"Jobs completed: {len(simulator.completed_jobs)}")
    print(f"Events processed: {len(history)}")
    print()

    for job in sorted(simulator.completed_jobs.values(), key=lambda j: j.job_id):
        print(
            f"Job {job.job_id:02d} | GPUs={job.gpu_count} | "
            f"arrival={job.arrival_time:6.1f}s | start={job.start_time:6.1f}s | "
            f"done={job.completion_time:6.1f}s | wait={job.waiting_time:6.1f}s"
        )


if __name__ == "__main__":
    main()
