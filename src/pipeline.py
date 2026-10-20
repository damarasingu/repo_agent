"""
End-to-End Pipeline Module.

Orchestrates:
Face Scan Input -> Face Detection/Encoding -> Web/Social Media Search -> Face Verification -> Blockchain Anchoring -> On-Chain Re-verification.
"""

import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

from src.face_engine import FaceEngine, FaceDetectionResult
from src.web_search import SocialMediaSearchEngine, DiscoveredPost
from src.blockchain_engine import BlockchainEngine, TransactionRecord, Block, VerificationProof


@dataclass
class PipelineResult:
    status: str  # "SUCCESS", "NO_FACE_DETECTED", "NO_MATCH_FOUND", "ERROR"
    input_image_path: str
    faces_detected: int
    target_face: Optional[Dict[str, Any]]
    matched_post: Optional[Dict[str, Any]]
    blockchain_tx: Optional[Dict[str, Any]]
    blockchain_block: Optional[Dict[str, Any]]
    verification_proof: Optional[Dict[str, Any]]
    execution_time_seconds: float
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FaceRecognitionBlockchainPipeline:
    """End-to-end verification pipeline."""

    def __init__(self, tolerance: float = 0.6, db_file: str = "blockchain_db.json"):
        self.face_engine = FaceEngine(tolerance=tolerance)
        self.search_engine = SocialMediaSearchEngine(face_engine=self.face_engine)
        self.blockchain_engine = BlockchainEngine(mode="local", db_file=db_file)

    def run(
        self,
        image_path: str,
        search_query: str = "profile face scan",
        similarity_threshold: float = 50.0,
        max_search_results: int = 8
    ) -> PipelineResult:
        """
        Run the complete pipeline end-to-end.
        """
        start_time = time.time()

        # Step 1: Detect and encode face from input image
        try:
            detected_faces = self.face_engine.detect_and_encode_faces(image_path)
        except Exception as e:
            return PipelineResult(
                status="ERROR",
                input_image_path=image_path,
                faces_detected=0,
                target_face=None,
                matched_post=None,
                blockchain_tx=None,
                blockchain_block=None,
                verification_proof=None,
                execution_time_seconds=round(time.time() - start_time, 2),
                message=f"Failed to read image or detect faces: {str(e)}"
            )

        if not detected_faces:
            return PipelineResult(
                status="NO_FACE_DETECTED",
                input_image_path=image_path,
                faces_detected=0,
                target_face=None,
                matched_post=None,
                blockchain_tx=None,
                blockchain_block=None,
                verification_proof=None,
                execution_time_seconds=round(time.time() - start_time, 2),
                message="No faces could be identified in the input image."
            )

        primary_face = detected_faces[0]

        # Step 2: Search Web & Social Media for matching content
        matches = self.search_engine.search_and_verify(
            target_face=primary_face,
            query=search_query,
            max_results=max_search_results,
            similarity_threshold=similarity_threshold
        )

        if not matches:
            return PipelineResult(
                status="NO_MATCH_FOUND",
                input_image_path=image_path,
                faces_detected=len(detected_faces),
                target_face={
                    "face_id": primary_face.face_id,
                    "bounding_box": primary_face.bounding_box,
                    "encoding_hash": primary_face.encoding_hash
                },
                matched_post=None,
                blockchain_tx=None,
                blockchain_block=None,
                verification_proof=None,
                execution_time_seconds=round(time.time() - start_time, 2),
                message="No social media / web posts matched the face scan above threshold."
            )

        best_match = matches[0]

        # Step 3: Anchor matching post & face scan onto the Blockchain
        tx_record, mined_block = self.blockchain_engine.anchor_match(
            image_hash=best_match.image_hash,
            face_encoding_hash=best_match.face_encoding_hash,
            post_url=best_match.post_url,
            platform=best_match.platform,
            author=best_match.author,
            similarity_score=best_match.face_similarity_score,
            metadata=best_match.raw_metadata
        )

        # Step 4: Re-verify against on-chain record to guarantee tamper-proof integrity
        proof = self.blockchain_engine.verify_match(
            tx_hash=tx_record.tx_hash,
            image_hash=best_match.image_hash,
            face_encoding_hash=best_match.face_encoding_hash,
            post_url=best_match.post_url,
            platform=best_match.platform
        )

        elapsed = round(time.time() - start_time, 2)

        return PipelineResult(
            status="SUCCESS",
            input_image_path=image_path,
            faces_detected=len(detected_faces),
            target_face={
                "face_id": primary_face.face_id,
                "bounding_box": primary_face.bounding_box,
                "encoding_hash": primary_face.encoding_hash
            },
            matched_post=best_match.to_dict(),
            blockchain_tx=tx_record.to_dict(),
            blockchain_block=mined_block.to_dict(),
            verification_proof=asdict(proof),
            execution_time_seconds=elapsed,
            message="End-to-end pipeline executed successfully: Face Scan -> Web Match -> Blockchain Verification."
        )

    def verify_on_chain(
        self,
        tx_hash: str,
        image_hash: str,
        face_encoding_hash: str,
        post_url: str,
        platform: str
    ) -> VerificationProof:
        """Standalone verification of record against on-chain state."""
        return self.blockchain_engine.verify_match(
            tx_hash=tx_hash,
            image_hash=image_hash,
            face_encoding_hash=face_encoding_hash,
            post_url=post_url,
            platform=platform
        )
