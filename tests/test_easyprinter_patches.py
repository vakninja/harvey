"""Tests for the easyprinter fork patches.

See the marketing repo: docs/superpowers/plans/2026-10-16-phase1a-harvey-engine.md
"""
import os
import tempfile

import pytest
import pytest_asyncio

import harvey.brain as brain_mod
from harvey.brain import Brain
from harvey.config import (
    ChannelsConfig,
    EmailChannelConfig,
    EnvConfig,
    HarveyConfig,
    ICPConfig,
    PersonaConfig,
    ProductConfig,
)
from harvey.models.prospect import Prospect
from harvey.state import StateManager

pytestmark = pytest.mark.asyncio


def _config(**email) -> HarveyConfig:
    return HarveyConfig(
        persona=PersonaConfig(
            name="עמית", company="Easy Printer", role="בעלים",
            email="outreach@easyprinter.co.il", linkedin="", tone="חם וישיר",
        ),
        product=ProductConfig(
            name="Easy Printer", description="תוכנה לבתי דפוס", pricing="302 ש\"ח",
            key_benefits=["פנקסים"], objection_responses={},
        ),
        icp=ICPConfig(industries=["print-shop"], company_size="1-10", titles=["בעלים"], geography=["Israel"]),
        channels=ChannelsConfig(email=EmailChannelConfig(provider="smtp", **email)),
    )


class FakeBrain:
    """Real prompt loading (from a monkeypatched PROMPTS_DIR), fake model calls."""

    load_prompt = Brain.load_prompt
    has_prompt = Brain.has_prompt

    def __init__(self, result=None):
        self.result = result
        self.prompts = []

    def load_skills_for_agent(self, agent):
        return ""

    async def think_json(self, prompt, **kw):
        self.prompts.append(prompt)
        return self.result


@pytest_asyncio.fixture
async def state():
    with tempfile.TemporaryDirectory() as d:
        sm = StateManager(os.path.join(d, "t.db"))
        await sm.init_db()
        yield sm


THREE = [
    {"step": 1, "subject": "a", "body": "b", "delay_days": 0},
    {"step": 2, "subject": "c", "body": "d", "delay_days": 3},
    {"step": 3, "subject": "e", "body": "f", "delay_days": 4},
]


# ── P1 + P2: max_sequence_steps and prompt override files ──


def test_max_sequence_steps_default_and_bounds():
    assert EmailChannelConfig().max_sequence_steps == 3
    assert EmailChannelConfig(max_sequence_steps=1).max_sequence_steps == 1
    with pytest.raises(Exception):
        EmailChannelConfig(max_sequence_steps=0)
    with pytest.raises(Exception):
        EmailChannelConfig(max_sequence_steps=6)


async def test_writer_parse_keeps_only_capped_steps(state):
    from harvey.agents.writer import Writer

    w = Writer(FakeBrain(), state, _config(max_sequence_steps=1))
    steps = w._parse_sequence(THREE)
    assert [s.step for s in steps] == [1]
    w3 = Writer(FakeBrain(), state, _config())
    assert [s.step for s in w3._parse_sequence(THREE)] == [1, 2, 3]


async def test_writer_sequence_prompt_asks_for_one_email(state, monkeypatch, tmp_path):
    from harvey.agents.writer import Writer

    monkeypatch.setattr(brain_mod, "PROMPTS_DIR", tmp_path)  # no override files
    brain = FakeBrain(result=THREE)
    w = Writer(brain, state, _config(max_sequence_steps=1))
    w.skills = ""
    steps = await w._write_sequence([Prospect(first_name="דפוס", last_name="", title="בעלים", company="דפוס")])
    assert len(steps) == 1
    p = brain.prompts[0]
    assert "ONE email" in p
    assert "Email 2" not in p and '"step": 2' not in p


async def test_writer_sequence_prompt_override_file(state, monkeypatch, tmp_path):
    from harvey.agents.writer import Writer

    (tmp_path / "writer_sequence.md").write_text(
        "כתוב {{steps}} מייל אחד עבור:\n{{prospect_summary}}\nRETURN JSON", encoding="utf-8"
    )
    monkeypatch.setattr(brain_mod, "PROMPTS_DIR", tmp_path)
    brain = FakeBrain(result=THREE[:1])
    w = Writer(brain, state, _config(max_sequence_steps=1))
    w.skills = ""
    await w._write_sequence([Prospect(first_name="דפוס הברון", last_name="", title="בעלים", company="דפוס הברון")])
    p = brain.prompts[0]
    assert "כתוב 1 מייל אחד" in p and "דפוס הברון" in p
    assert "Write a" not in p  # upstream inline text replaced


async def test_writer_personal_prompt_override_file(state, monkeypatch, tmp_path):
    from harvey.agents.writer import Writer

    (tmp_path / "writer_personal.md").write_text("עובדות:\n{{facts}}\nכתוב מייל פרסומת אחד.", encoding="utf-8")
    monkeypatch.setattr(brain_mod, "PROMPTS_DIR", tmp_path)
    brain = FakeBrain(result={"subject": "פרסומת: x", "body": "y"})
    w = Writer(brain, state, _config(max_sequence_steps=1))
    w.skills = ""
    draft = await w._write_personal_email(
        Prospect(first_name="דפוס הברון", last_name="", title="בעלים", company="דפוס הברון")
    )
    assert draft == {"subject": "פרסומת: x", "body": "y"}
    p = brain.prompts[0]
    assert "כתוב מייל פרסומת אחד" in p and "- Company: דפוס הברון" in p
    assert "Under 75 words" not in p


async def test_writer_personal_prompt_upstream_text_without_override(state, monkeypatch, tmp_path):
    from harvey.agents.writer import Writer

    monkeypatch.setattr(brain_mod, "PROMPTS_DIR", tmp_path)
    brain = FakeBrain(result={"subject": "s", "body": "b"})
    w = Writer(brain, state, _config())
    w.skills = ""
    await w._write_personal_email(Prospect(first_name="A", last_name="B", title="T", company="C"))
    assert "Under 75 words" in brain.prompts[0]


def test_brain_has_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr(brain_mod, "PROMPTS_DIR", tmp_path)
    b = Brain.__new__(Brain)
    assert b.has_prompt("writer_personal") is False
    (tmp_path / "writer_personal.md").write_text("x", encoding="utf-8")
    assert b.has_prompt("writer_personal") is True


# ── P3: Sender stages at most max_sequence_steps ──


class _FakeProvider:
    name = "fake"

    def is_configured(self):
        return True


async def test_sender_stages_only_capped_steps(state):
    from harvey.agents.sender import Sender
    from harvey.models.campaign import Campaign, EmailStep
    from harvey.models.company import Company

    cid = await state.add_company(Company(name="דפוס הברון", domain="habaron.co.il", industry="print-shop"))
    pid = await state.add_prospect(Prospect(
        company_id=cid, first_name="דפוס הברון", last_name="", title="בעלים",
        email="shop@habaron.co.il", email_status="risky", status="new",
        company="דפוס הברון", industry="print-shop",
    ))
    seq = [EmailStep(step=i, subject=f"s{i}", body=f"b{i}", delay_days=d) for i, d in ((1, 0), (2, 3), (3, 4))]
    camp = Campaign(id="c1", name="x", prospect_ids=[pid], sequence=seq, status="draft")
    await state.add_campaign(camp)
    s = Sender(FakeBrain(), state, _config(max_sequence_steps=1, send_to_risky=True), EnvConfig())
    s.provider = _FakeProvider()
    await s._stage_campaign_native(camp)
    rows = await state.get_outbox(status="pending_review", limit=50)
    assert [r["step"] for r in rows if r["prospect_id"] == pid] == [1]


# ── P4 + P5: HARVEY_DISABLE_SCOUT and print-shop OSM tags ──


async def test_disable_scout_env_turns_prospecting_into_idle(monkeypatch):
    from harvey.main import decide_next_action

    summary = {"prospects": {"new": 0}, "draft_campaigns": 0, "open_conversations": 0}
    monkeypatch.delenv("HARVEY_DISABLE_SCOUT", raising=False)
    assert await decide_next_action(None, None, _config(), summary=summary) == "prospect"
    monkeypatch.setenv("HARVEY_DISABLE_SCOUT", "1")
    assert await decide_next_action(None, None, _config(), summary=summary) == "idle"
    summary["prospects"]["new"] = 3
    assert await decide_next_action(None, None, _config(), summary=summary) == "write_campaign"


def test_osm_tag_map_knows_print_shops():
    from harvey.collectors.discover import OpenStreetMap

    assert OpenStreetMap.TAG_MAP["print"] == ['shop=copyshop', 'craft=printer', 'shop=printing']
    osm = OpenStreetMap.__new__(OpenStreetMap)
    assert osm._tags("print shop") == ['shop=copyshop', 'craft=printer', 'shop=printing']


# ── P6: Hebrew opt-out phrases are hard matches ──


@pytest.mark.parametrize("text", [
    "נא להסיר אותי מהרשימה",
    "הסירו אותי",
    "תפסיקו לשלוח לי מיילים",
    "לא מעוניין תודה",
    "אל תשלחו יותר",
    "הסר",
])
def test_hebrew_opt_out_phrases_are_hard_matches(text):
    from harvey.agents.handler import OPT_OUT_PATTERNS

    assert any(p in text.lower() for p in OPT_OUT_PATTERNS), text


def test_hebrew_interest_is_not_an_opt_out():
    from harvey.agents.handler import OPT_OUT_PATTERNS

    assert not any(p in "כן, אשמח לקבל פרטים" for p in OPT_OUT_PATTERNS)
