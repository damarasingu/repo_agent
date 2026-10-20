"""
Main Entry Point & Command Line Interface for Face Recognition & Blockchain Verification Pipeline.
"""

import sys
import os
import json
import argparse
from typing import Dict, Any

# Ensure current workspace is on Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline import FaceRecognitionBlockchainPipeline
from src.blockchain_engine import BlockchainEngine
from samples.generate_sample_faces import generate_sample_images

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.json import JSON
    from rich import print as rprint
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False


def print_banner():
    banner_text = """
========================================================================
   FACE RECOGNITION & BLOCKCHAIN VERIFICATION PIPELINE
   Face Scan Input -> Web/Social Search -> Blockchain Verification
========================================================================
    """
    if HAS_RICH:
        console.print(Panel.fit(
            "[bold cyan]FACE RECOGNITION & BLOCKCHAIN VERIFICATION PIPELINE[/bold cyan]\n"
            "[yellow]Face Scan Input → Web/Social Search → Immutable Blockchain Verification[/yellow]",
            border_style="cyan"
        ))
    else:
        print(banner_text)


def run_scan(args):
    """Execute scan pipeline command."""
    print_banner()

    image_path = args.image
    query = args.query
    threshold = args.threshold

    # Ensure sample image exists if user picked sample default
    if not os.path.exists(image_path):
        print(f"Sample image '{image_path}' not found. Generating sample face assets...")
        generate_sample_images()

    print(f"[*] Starting Pipeline...")
    print(f"    - Input Image: {image_path}")
    print(f"    - Search Query: '{query}'")
    print(f"    - Match Threshold: {threshold}%")

    pipeline = FaceRecognitionBlockchainPipeline(tolerance=0.6)
    result = pipeline.run(
        image_path=image_path,
        search_query=query,
        similarity_threshold=threshold
    )

    res_dict = result.to_dict()

    if HAS_RICH:
        if result.status == "SUCCESS":
            console.print("\n[bold green]✓ PIPELINE COMPLETED SUCCESSFULLY![/bold green]\n")

            # Face Details
            face_table = Table(title="1. Face Identification", show_header=True, header_style="bold magenta")
            face_table.add_column("Property", style="dim")
            face_table.add_column("Value")
            face_table.add_row("Input File", result.input_image_path)
            face_table.add_row("Faces Detected", str(result.faces_detected))
            face_table.add_row("Target Face ID", result.target_face.get("face_id", "N/A"))
            face_table.add_row("Encoding SHA-256", result.target_face.get("encoding_hash", "")[:24] + "...")
            console.print(face_table)

            # Match Details
            post = result.matched_post
            post_table = Table(title="2. Web & Social Media Search Match", show_header=True, header_style="bold cyan")
            post_table.add_column("Field", style="dim")
            post_table.add_column("Details")
            post_table.add_row("Platform", f"[bold yellow]{post.get('platform')}[/bold yellow]")
            post_table.add_row("Post URL", post.get("post_url"))
            post_table.add_row("Author", post.get("author"))
            post_table.add_row("Similarity Score", f"[bold green]{post.get('face_similarity_score')}%[/bold green]")
            post_table.add_row("Image SHA-256", post.get("image_hash")[:24] + "...")
            console.print(post_table)

            # Blockchain Details
            tx = result.blockchain_tx
            blk = result.blockchain_block
            chain_table = Table(title="3. Blockchain Anchoring & Verification Proof", show_header=True, header_style="bold green")
            chain_table.add_column("Blockchain Property", style="dim")
            chain_table.add_column("Record Value")
            chain_table.add_row("Transaction Hash", f"[cyan]{tx.get('tx_hash')}[/cyan]")
            chain_table.add_row("Payload SHA-256", tx.get("payload_hash"))
            chain_table.add_row("Block Number", str(blk.get("index")))
            chain_table.add_row("Block Hash", blk.get("block_hash")[:24] + "...")
            chain_table.add_row("Merkle Root", blk.get("merkle_root")[:24] + "...")
            chain_table.add_row("On-Chain Status", "[bold green]VERIFIED & TAMPER-EVIDENT[/bold green]")
            console.print(chain_table)

            proof = result.verification_proof
            console.print(Panel(
                f"[bold green]✓ RE-VERIFICATION PROOF PASSED[/bold green]\n"
                f"{proof.get('verification_message')}\n"
                f"Payload Hash Match: [bold]{proof.get('expected_payload_hash')}[/bold]",
                title="On-Chain Tamper Audit",
                border_style="green"
            ))
        else:
            console.print(f"[bold red]Pipeline Status: {result.status}[/bold red]")
            console.print(f"Message: {result.message}")

    else:
        print(json.dumps(res_dict, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(res_dict, f, indent=2)
        print(f"\n[+] Full pipeline result exported to: {args.output}")


def run_verify(args):
    """Execute on-chain verification command."""
    print_banner()

    engine = BlockchainEngine(mode="local")
    proof = engine.verify_match(
        tx_hash=args.tx_hash,
        image_hash=args.image_hash,
        face_encoding_hash=args.encoding_hash,
        post_url=args.post_url,
        platform=args.platform
    )

    proof_dict = {
        "is_valid": proof.is_valid,
        "tamper_detected": proof.tamper_detected,
        "block_number": proof.block_number,
        "block_hash": proof.block_hash,
        "merkle_root": proof.merkle_root,
        "expected_payload_hash": proof.expected_payload_hash,
        "actual_payload_hash": proof.actual_payload_hash,
        "message": proof.verification_message
    }

    if HAS_RICH:
        style = "green" if proof.is_valid else "red"
        status_text = "VERIFIED IMMUTABLE RECORD" if proof.is_valid else "TAMPER DETECTED / RECORD INVALID"
        console.print(Panel(
            f"[bold {style}]{status_text}[/bold {style}]\n\n"
            f"Message: {proof.verification_message}\n"
            f"Block Index: {proof.block_number}\n"
            f"Block Hash: {proof.block_hash}\n"
            f"Expected Payload Hash: {proof.expected_payload_hash}\n"
            f"On-Chain Payload Hash: {proof.actual_payload_hash}",
            title="Blockchain Re-verification Result",
            border_style=style
        ))
    else:
        print(json.dumps(proof_dict, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Face Recognition & Blockchain Verification Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Run end-to-end pipeline on input face scan")
    scan_parser.add_argument("--image", type=str, default="samples/input_face.jpg", help="Path to input face image")
    scan_parser.add_argument("--query", type=str, default="Namyad Hingra", help="Search query for web / social media search")
    scan_parser.add_argument("--threshold", type=float, default=50.0, help="Face similarity threshold percentage (0-100)")
    scan_parser.add_argument("--output", type=str, help="Path to save JSON output result")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify record against blockchain ledger")
    verify_parser.add_argument("--tx-hash", type=str, required=True, help="Blockchain transaction hash")
    verify_parser.add_argument("--image-hash", type=str, required=True, help="SHA-256 hash of matched post image")
    verify_parser.add_argument("--encoding-hash", type=str, required=True, help="SHA-256 hash of face scan encoding")
    verify_parser.add_argument("--post-url", type=str, required=True, help="Discovered post URL")
    verify_parser.add_argument("--platform", type=str, default="Web Platform", help="Social media platform")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run automated demonstration")

    args = parser.parse_args()

    if args.command == "scan":
        run_scan(args)
    elif args.command == "verify":
        run_verify(args)
    elif args.command == "demo" or len(sys.argv) == 1:
        args.image = "samples/input_face.jpg"
        args.query = "Namyad Hingra"
        args.threshold = 50.0
        args.output = "pipeline_output.json"
        run_scan(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
