"""Шаги сборки датасета."""

from querulus.dataset.steps.claims import load_claims
from querulus.dataset.steps.enrich import enrich_dataset
from querulus.dataset.steps.payments import load_claims_payments
from querulus.dataset.steps.pretensions import load_pretensions
from querulus.dataset.steps.targets import build_targets
from querulus.dataset.steps.victim import load_victim
