from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.meeting_summarizer.config import MeetingConfigError, load_meeting_config
from src.meeting_summarizer.schemas.artifacts import MeetingSummaryArtifact, SegmentRecord
from src.meeting_summarizer.utils.ids import (
    build_candidate_id,
    build_case_id,
    build_meeting_id,
    build_segment_id,
)


class ConfigIdsSchemaTest(unittest.TestCase):
    def test_schema_validation(self) -> None:
        seg = SegmentRecord(segment_id="seg_a", start_sec=0.0, end_sec=10.0, text="hello")
        artifact = MeetingSummaryArtifact(
            meeting_id="mtg_2026-05-04_test",
            segments=[seg],
            candidates=[],
            cases=[],
            final_summary="summary",
        )
        self.assertEqual(artifact.segments[0].segment_id, "seg_a")

    def test_config_missing_file_raises_clear_error(self) -> None:
        with self.assertRaises(MeetingConfigError) as ctx:
            load_meeting_config("configs/not-found.yaml")
        self.assertIn("config file not found", str(ctx.exception))

    def test_config_load_success(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "meeting.yaml"
            p.write_text(
                """
artifacts:
  version: v1
  output_dir: data/processed/meeting
id_rules:
  meeting_prefix: mtg
  segment_prefix: seg
  candidate_prefix: cand
  case_prefix: case
""".strip(),
                encoding="utf-8",
            )
            cfg = load_meeting_config(p)
            self.assertEqual(cfg["artifacts"]["version"], "v1")

    def test_id_contracts(self) -> None:
        meeting_id = build_meeting_id(title="War Game Brief", date_iso="2026-05-04")
        segment_id = build_segment_id(meeting_id=meeting_id, index=3)
        candidate_id = build_candidate_id(meeting_id=meeting_id, segment_ids=[segment_id])
        case_id = build_case_id(meeting_id=meeting_id, candidate_ids=[candidate_id])

        self.assertTrue(meeting_id.startswith("mtg_2026-05-04_"))
        self.assertEqual(segment_id, f"seg_{meeting_id}_0003")
        self.assertTrue(candidate_id.startswith(f"cand_{meeting_id}_"))
        self.assertTrue(case_id.startswith(f"case_{meeting_id}_"))


if __name__ == "__main__":
    unittest.main()
