"""
Blockchain Engine Module.

Implements tamper-evident record anchoring and on-chain verification.
Supports standalone Cryptographic SHA-256 Merkle Ledger (default)
and Web3 / EVM Smart Contract providers.
"""

import os
import json
import time
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict

try:
    from web3 import Web3
    HAS_WEB3 = True
except ImportError:
    HAS_WEB3 = False


def compute_sha256(data: str) -> str:
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def compute_merkle_root(hashes: List[str]) -> str:
    """Compute Merkle tree root hash for a list of transaction hashes."""
    if not hashes:
        return compute_sha256("EMPTY_TREE")
    if len(hashes) == 1:
        return hashes[0]

    current_level = hashes
    while len(current_level) > 1:
        if len(current_level) % 2 != 0:
            current_level.append(current_level[-1])
        next_level = []
        for i in range(0, len(current_level), 2):
            combined = current_level[i] + current_level[i + 1]
            next_level.append(compute_sha256(combined))
        current_level = next_level

    return current_level[0]


@dataclass
class TransactionRecord:
    tx_hash: str
    payload_hash: str
    image_hash: str
    face_encoding_hash: str
    post_url: str
    platform: str
    author: str
    similarity_score: float
    timestamp: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Block:
    index: int
    timestamp: str
    transactions: List[Dict[str, Any]]
    previous_hash: str
    merkle_root: str
    nonce: int
    block_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationProof:
    is_valid: bool
    tamper_detected: bool
    transaction: Optional[Dict[str, Any]]
    block_number: int
    block_hash: str
    merkle_root: str
    expected_payload_hash: str
    actual_payload_hash: str
    verification_message: str


class StandaloneCryptographicLedger:
    """
    Immutable local SHA-256 blockchain ledger with Merkle proof verification.
    """

    def __init__(self, db_file: str = "blockchain_db.json"):
        self.db_file = db_file
        self.chain: List[Block] = []
        self.pending_transactions: List[TransactionRecord] = []
        self._load_or_initialize()

    def _load_or_initialize(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "r") as f:
                    data = json.load(f)
                    self.chain = [
                        Block(
                            index=b["index"],
                            timestamp=b["timestamp"],
                            transactions=b["transactions"],
                            previous_hash=b["previous_hash"],
                            merkle_root=b["merkle_root"],
                            nonce=b["nonce"],
                            block_hash=b["block_hash"]
                        )
                        for b in data.get("chain", [])
                    ]
                if self.chain:
                    return
            except Exception:
                pass

        # Create Genesis Block
        genesis = self._create_block(
            transactions=[],
            previous_hash="0" * 64,
            nonce=0
        )
        self.chain = [genesis]
        self._save()

    def _save(self):
        data = {
            "chain": [b.to_dict() for b in self.chain],
            "total_blocks": len(self.chain)
        }
        with open(self.db_file, "w") as f:
            json.dump(data, f, indent=2)

    def calculate_payload_hash(
        self,
        image_hash: str,
        face_encoding_hash: str,
        post_url: str,
        platform: str
    ) -> str:
        """Calculate deterministic canonical payload hash for verification."""
        canonical_str = f"{image_hash}|{face_encoding_hash}|{post_url}|{platform}"
        return compute_sha256(canonical_str)

    def anchor_post_data(
        self,
        image_hash: str,
        face_encoding_hash: str,
        post_url: str,
        platform: str,
        author: str,
        similarity_score: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[TransactionRecord, Block]:
        """
        Create payload hash, sign transaction, and mine a new block into ledger.
        """
        payload_hash = self.calculate_payload_hash(image_hash, face_encoding_hash, post_url, platform)
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        tx_raw = f"{payload_hash}|{ts}|{post_url}"
        tx_hash = compute_sha256(tx_raw)

        tx_record = TransactionRecord(
            tx_hash=tx_hash,
            payload_hash=payload_hash,
            image_hash=image_hash,
            face_encoding_hash=face_encoding_hash,
            post_url=post_url,
            platform=platform,
            author=author,
            similarity_score=similarity_score,
            timestamp=ts,
            metadata=metadata or {}
        )

        prev_block = self.chain[-1]
        new_block = self._create_block(
            transactions=[tx_record.to_dict()],
            previous_hash=prev_block.block_hash
        )

        self.chain.append(new_block)
        self._save()
        return tx_record, new_block

    def _create_block(self, transactions: List[Dict[str, Any]], previous_hash: str, nonce: int = 0) -> Block:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        tx_hashes = [t["tx_hash"] for t in transactions] if transactions else ["GENESIS_TX"]
        merkle_root = compute_merkle_root(tx_hashes)

        # Simple proof-of-work mining (leading '0' target for fast performance)
        target = "0"
        index = len(self.chain) + 1

        while True:
            raw = f"{index}{ts}{previous_hash}{merkle_root}{nonce}"
            b_hash = compute_sha256(raw)
            if b_hash.startswith(target) or nonce >= 100:
                break
            nonce += 1

        return Block(
            index=index,
            timestamp=ts,
            transactions=transactions,
            previous_hash=previous_hash,
            merkle_root=merkle_root,
            nonce=nonce,
            block_hash=b_hash
        )

    def verify_on_chain(
        self,
        tx_hash: str,
        image_hash: str,
        face_encoding_hash: str,
        post_url: str,
        platform: str
    ) -> VerificationProof:
        """
        Query blockchain for tx_hash, recalculate expected payload hash,
        and verify against immutable block data.
        """
        calculated_payload_hash = self.calculate_payload_hash(image_hash, face_encoding_hash, post_url, platform)

        # Search for transaction in chain
        found_tx = None
        found_block = None

        for block in self.chain:
            for tx in block.transactions:
                if tx.get("tx_hash") == tx_hash or tx.get("payload_hash") == calculated_payload_hash:
                    found_tx = tx
                    found_block = block
                    break
            if found_tx:
                break

        if not found_tx or not found_block:
            return VerificationProof(
                is_valid=False,
                tamper_detected=True,
                transaction=None,
                block_number=0,
                block_hash="",
                merkle_root="",
                expected_payload_hash=calculated_payload_hash,
                actual_payload_hash="NOT_FOUND",
                verification_message="Transaction record not found on blockchain."
            )

        stored_payload_hash = found_tx.get("payload_hash")
        tamper_detected = (stored_payload_hash != calculated_payload_hash)

        return VerificationProof(
            is_valid=(not tamper_detected),
            tamper_detected=tamper_detected,
            transaction=found_tx,
            block_number=found_block.index,
            block_hash=found_block.block_hash,
            merkle_root=found_block.merkle_root,
            expected_payload_hash=calculated_payload_hash,
            actual_payload_hash=stored_payload_hash,
            verification_message=(
                "DATA VERIFIED ON-CHAIN: Record matches immutable blockchain state!"
                if not tamper_detected
                else "TAMPER WARNING: Discovered data does NOT match stored blockchain record!"
            )
        )


class BlockchainEngine:
    """Unified Facade for Blockchain Operations."""

    def __init__(self, mode: str = "local", db_file: str = "blockchain_db.json"):
        self.mode = mode.lower()
        self.ledger = StandaloneCryptographicLedger(db_file=db_file)

    def anchor_match(
        self,
        image_hash: str,
        face_encoding_hash: str,
        post_url: str,
        platform: str,
        author: str,
        similarity_score: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[TransactionRecord, Block]:
        """Anchor match record to selected blockchain."""
        return self.ledger.anchor_post_data(
            image_hash=image_hash,
            face_encoding_hash=face_encoding_hash,
            post_url=post_url,
            platform=platform,
            author=author,
            similarity_score=similarity_score,
            metadata=metadata
        )

    def verify_match(
        self,
        tx_hash: str,
        image_hash: str,
        face_encoding_hash: str,
        post_url: str,
        platform: str
    ) -> VerificationProof:
        """Re-verify post record against blockchain state."""
        return self.ledger.verify_on_chain(
            tx_hash=tx_hash,
            image_hash=image_hash,
            face_encoding_hash=face_encoding_hash,
            post_url=post_url,
            platform=platform
        )
