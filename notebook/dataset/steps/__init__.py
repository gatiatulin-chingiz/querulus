"""Шаги сборки датасета."""

from dataset.steps.victim import load_victim
from dataset.steps.claims import load_claims
from dataset.steps.payments import load_claims_payments
from dataset.steps.pretensions import load_pretensions
from dataset.steps.enrich import enrich_dataset
from dataset.steps.targets import build_targets
