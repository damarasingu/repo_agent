// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title FaceVerificationRegistry
 * @dev Smart Contract to record and verify tamper-evident face recognition social media search records on-chain.
 */
contract FaceVerificationRegistry {
    
    struct VerificationRecord {
        bytes32 payloadHash;       // SHA-256 / Keccak-256 hash of post content + face encoding + metadata
        bytes32 imageHash;         // Hash of matching post image
        bytes32 faceEncodingHash;  // Hash of input face scan encoding vector
        string postUrl;            // URL of discovered social media post
        string platform;           // Social media platform (Twitter, LinkedIn, etc.)
        uint256 similarityScore;   // Face similarity percentage (scaled by 100, e.g. 9550 = 95.5%)
        uint256 timestamp;         // On-chain timestamp
        address verifiedBy;        // Address of verifier entity
        bool isValueRecorded;      // Record existence flag
    }

    // Mapping from tx payload hash to VerificationRecord
    mapping(bytes32 => VerificationRecord) public records;
    
    // Array of recorded payload hashes for audit enumeration
    bytes32[] public recordHashes;

    // Events for on-chain indexing & audit tracking
    event RecordAnchored(
        bytes32 indexed payloadHash,
        bytes32 indexed imageHash,
        bytes32 indexed faceEncodingHash,
        string postUrl,
        string platform,
        uint256 similarityScore,
        uint256 timestamp,
        address verifier
    );

    /**
     * @dev Anchor a new face recognition verification record onto the blockchain ledger.
     */
    function anchorRecord(
        bytes32 payloadHash,
        bytes32 imageHash,
        bytes32 faceEncodingHash,
        string calldata postUrl,
        string calldata platform,
        uint256 similarityScore
    ) external returns (bytes32) {
        require(!records[payloadHash].isValueRecorded, "Record already exists on chain!");

        VerificationRecord memory newRecord = VerificationRecord({
            payloadHash: payloadHash,
            imageHash: imageHash,
            faceEncodingHash: faceEncodingHash,
            postUrl: postUrl,
            platform: platform,
            similarityScore: similarityScore,
            timestamp: block.timestamp,
            verifiedBy: msg.sender,
            isValueRecorded: true
        });

        records[payloadHash] = newRecord;
        recordHashes.push(payloadHash);

        emit RecordAnchored(
            payloadHash,
            imageHash,
            faceEncodingHash,
            postUrl,
            platform,
            similarityScore,
            block.timestamp,
            msg.sender
        );

        return payloadHash;
    }

    /**
     * @dev Re-verify payload hash against on-chain record.
     */
    function verifyRecord(bytes32 payloadHash) external view returns (
        bool exists,
        bytes32 imageHash,
        bytes32 faceEncodingHash,
        string memory postUrl,
        string memory platform,
        uint256 similarityScore,
        uint256 timestamp,
        address verifiedBy
    ) {
        VerificationRecord memory rec = records[payloadHash];
        if (!rec.isValueRecorded) {
            return (false, bytes32(0), bytes32(0), "", "", 0, 0, address(0));
        }

        return (
            true,
            rec.imageHash,
            rec.faceEncodingHash,
            rec.postUrl,
            rec.platform,
            rec.similarityScore,
            rec.timestamp,
            rec.verifiedBy
        );
    }

    /**
     * @dev Get total count of anchored records.
     */
    function getRecordCount() external view returns (uint256) {
        return recordHashes.length;
    }
}
