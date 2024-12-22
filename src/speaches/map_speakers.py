from dataclasses import dataclass, asdict
import json
from typing import List, Optional

@dataclass
class TranscriptionSegment:
    id: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

@dataclass
class DiarizationSegment:
    speaker: str
    start: float
    end: float

def calculate_overlap(trans_seg: TranscriptionSegment, diar_seg: DiarizationSegment) -> float:
    """Calculate the temporal overlap between a transcription and diarization segment."""
    overlap_start = max(trans_seg.start, diar_seg.start)
    overlap_end = min(trans_seg.end, diar_seg.end)
    return max(0, overlap_end - overlap_start)

def map_speakers_to_segments(
    transcription_segments: List[TranscriptionSegment],
    diarization_segments: List[DiarizationSegment]
) -> str:
    """
    Maps speakers to transcription segments and returns JSON string.
    """
    result = []
    
    for trans_seg in transcription_segments:
        max_overlap = 0
        best_speaker = None
        
        # Find diarization segment with maximum overlap
        for diar_seg in diarization_segments:
            overlap = calculate_overlap(trans_seg, diar_seg)
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = diar_seg.speaker
        
        # Create new segment with assigned speaker
        new_segment = TranscriptionSegment(
            id=trans_seg.id,
            start=trans_seg.start,
            end=trans_seg.end,
            text=trans_seg.text,
            speaker=best_speaker
        )
        result.append(new_segment)
    
    # Convert the result to a list of dictionaries
    result_json = [asdict(segment) for segment in result]
    
    # Return JSON string with proper encoding for Cyrillic characters
    return json.dumps(result_json, ensure_ascii=False, indent=2)