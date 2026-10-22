"""Celery application configuration."""
from celery import Celery
from api.core.config import settings

celery_app = Celery(
    "waste_api",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "api.workers.tasks.predict_task",
        "api.workers.tasks.capture_task",
    ],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=settings.CELERY_TASK_TRACK_STARTED,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
    task_default_queue="prediction",
    task_routes={"api.workers.tasks.capture_task.run_capture_task": {"queue": "capture"}},
    broker_connection_timeout=3,
    broker_transport_options={"socket_timeout": 3, "socket_connect_timeout": 3, "visibility_timeout": 1800},
    beat_schedule={},  # Periodic tasks will be added later
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["api.workers.tasks"])


@celery_app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f"Request: {self.request!r}")
