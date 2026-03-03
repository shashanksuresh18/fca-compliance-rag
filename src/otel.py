"""
otel.py — Wave 3: OpenTelemetry Observability for Distributed Tracing.

WHY OPENTELEMETRY?
━━━━━━━━━━━━━━━━━━
In a multi-stage RAG pipeline (LangGraph), finding the "bottleneck" or 
the point of failure requires seeing the full execution span.
  - Did retrieval take too long?
  - Did the LLM hang?
  - Did the evidence grader fail to match?

This module provides the enterprise-grade plumbing for distributed tracing
compatible with Azure Monitor, Jaeger, or Honeycomb.
"""

import os
import logging
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from src.config import settings

logger = logging.getLogger(__name__)

def setup_otel(app: FastAPI):
    """
    Initialize OpenTelemetry tracing for the FastAPI app and internal logic.
    """
    # 1. Resource identity
    resource = Resource(attributes={
        SERVICE_NAME: "fca-compliance-rag-service",
        "env": "development",
        "version": "1.0.0-p4"
    })

    # 2. Tracer Provider
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    # 3. Exporter (Console for local, OTLP for Azure/Jaeger)
    # In a real bank setup, this would point to Azure Monitor (App Insights)
    # via OTLPExporter.
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)

    # 4. Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)
    
    logger.info("OpenTelemetry instrumentation initialized.")

def get_tracer():
    """Returns the global tracer for manual instrumentation of nodes."""
    return trace.get_tracer(__name__)
