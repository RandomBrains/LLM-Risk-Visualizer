"""
Blockchain-Based Audit Trail and Immutable Record System
Implements blockchain technology for tamper-proof audit logging and compliance
"""

import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import pickle
import base64
from pathlib import Path
import sqlite3
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
import secrets
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionType(Enum):
    """Types of blockchain transactions"""
    RISK_ASSESSMENT = "risk_assessment"
    USER_ACTION = "user_action"
    MODEL_UPDATE = "model_update"
    DATA_ACCESS = "data_access"
    SYSTEM_EVENT = "system_event"
    COMPLIANCE_CHECK = "compliance_check"
    SECURITY_EVENT = "security_event"
    CONFIGURATION_CHANGE = "configuration_change"

class BlockValidationStatus(Enum):
    """Block validation status"""
    VALID = "valid"
    INVALID = "invalid"
    PENDING = "pending"
    CORRUPTED = "corrupted"

@dataclass
class Transaction:
    """Blockchain transaction with audit information"""
    transaction_id: str
    transaction_type: TransactionType
    timestamp: datetime
    user_id: str
    action: str
    data_hash: str
    metadata: Dict[str, Any]
    digital_signature: Optional[str] = None
    compliance_flags: List[str] = None
    risk_level: str = "low"
    
    def __post_init__(self):
        if self.compliance_flags is None:
            self.compliance_flags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary"""
        return {
            'transaction_id': self.transaction_id,
            'transaction_type': self.transaction_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'action': self.action,
            'data_hash': self.data_hash,
            'metadata': self.metadata,
            'digital_signature': self.digital_signature,
            'compliance_flags': self.compliance_flags,
            'risk_level': self.risk_level
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        """Create transaction from dictionary"""
        return cls(
            transaction_id=data['transaction_id'],
            transaction_type=TransactionType(data['transaction_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            user_id=data['user_id'],
            action=data['action'],
            data_hash=data['data_hash'],
            metadata=data['metadata'],
            digital_signature=data.get('digital_signature'),
            compliance_flags=data.get('compliance_flags', []),
            risk_level=data.get('risk_level', 'low')
        )

@dataclass
class Block:
    """Blockchain block containing multiple transactions"""
    block_id: int
    previous_hash: str
    timestamp: datetime
    transactions: List[Transaction]
    merkle_root: str
    nonce: int
    block_hash: str
    validator: str
    difficulty: int = 4
    
    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_string = json.dumps({
            'block_id': self.block_id,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp.isoformat(),
            'transactions': [tx.to_dict() for tx in self.transactions],
            'merkle_root': self.merkle_root,
            'nonce': self.nonce,
            'validator': self.validator
        }, sort_keys=True)
        
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def calculate_merkle_root(self) -> str:
        """Calculate Merkle root of transactions"""
        if not self.transactions:
            return hashlib.sha256(b"").hexdigest()
        
        # Create list of transaction hashes
        tx_hashes = [
            hashlib.sha256(json.dumps(tx.to_dict(), sort_keys=True).encode()).hexdigest()
            for tx in self.transactions
        ]
        
        # Build Merkle tree
        while len(tx_hashes) > 1:
            next_level = []
            for i in range(0, len(tx_hashes), 2):
                if i + 1 < len(tx_hashes):
                    combined = tx_hashes[i] + tx_hashes[i + 1]
                else:
                    combined = tx_hashes[i] + tx_hashes[i]  # Duplicate if odd number
                
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())
            tx_hashes = next_level
        
        return tx_hashes[0]
    
    def mine_block(self, difficulty: int = 4) -> bool:
        """Mine block using proof-of-work"""
        target = "0" * difficulty
        self.nonce = 0
        
        while True:
            self.merkle_root = self.calculate_merkle_root()
            hash_attempt = self.calculate_hash()
            
            if hash_attempt.startswith(target):
                self.block_hash = hash_attempt
                logger.info(f"Block {self.block_id} mined with nonce {self.nonce}")
                return True
            
            self.nonce += 1
            
            # Prevent infinite loops in testing
            if self.nonce > 1000000:
                logger.warning(f"Mining timeout for block {self.block_id}")
                self.block_hash = hash_attempt
                return False
    
    def validate(self, previous_block_hash: str = None) -> BlockValidationStatus:
        """Validate block integrity"""
        try:
            # Check if block hash is correct
            calculated_hash = self.calculate_hash()
            if calculated_hash != self.block_hash:
                return BlockValidationStatus.INVALID
            
            # Check if previous hash matches
            if previous_block_hash and self.previous_hash != previous_block_hash:
                return BlockValidationStatus.INVALID
            
            # Validate Merkle root
            calculated_merkle = self.calculate_merkle_root()
            if calculated_merkle != self.merkle_root:
                return BlockValidationStatus.INVALID
            
            # Validate each transaction
            for transaction in self.transactions:
                if not self._validate_transaction(transaction):
                    return BlockValidationStatus.INVALID
            
            return BlockValidationStatus.VALID
            
        except Exception as e:
            logger.error(f"Block validation error: {e}")
            return BlockValidationStatus.CORRUPTED
    
    def _validate_transaction(self, transaction: Transaction) -> bool:
        """Validate individual transaction"""
        try:
            # Check required fields
            if not all([
                transaction.transaction_id,
                transaction.user_id,
                transaction.action,
                transaction.data_hash
            ]):
                return False
            
            # Validate timestamp (should be reasonable)
            now = datetime.now()
            if transaction.timestamp > now + timedelta(minutes=5):
                return False  # Future timestamp not allowed
            
            # Validate data hash format
            if len(transaction.data_hash) != 64:  # SHA-256 hash length
                return False
            
            return True
            
        except Exception:
            return False

class CryptographicManager:
    """Manages cryptographic operations for blockchain"""
    
    def __init__(self):
        self.private_key = None
        self.public_key = None
        self.key_pair_generated = False
    
    def generate_key_pair(self) -> Tuple[str, str]:
        """Generate RSA key pair for digital signatures"""
        try:
            # Generate private key
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Get public key
            self.public_key = self.private_key.public_key()
            
            # Serialize keys
            private_pem = self.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            self.key_pair_generated = True
            
            return private_pem.decode('utf-8'), public_pem.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Key generation error: {e}")
            return None, None
    
    def sign_data(self, data: str) -> Optional[str]:
        """Sign data with private key"""
        if not self.private_key:
            return None
        
        try:
            signature = self.private_key.sign(
                data.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return base64.b64encode(signature).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Signing error: {e}")
            return None
    
    def verify_signature(self, data: str, signature: str, public_key_pem: str) -> bool:
        """Verify digital signature"""
        try:
            public_key = load_pem_public_key(public_key_pem.encode('utf-8'))
            signature_bytes = base64.b64decode(signature.encode('utf-8'))
            
            public_key.verify(
                signature_bytes,
                data.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False
    
    def hash_data(self, data: Any) -> str:
        """Create SHA-256 hash of data"""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        elif isinstance(data, str):
            data_str = data
        else:
            data_str = str(data)
        
        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()

class AuditBlockchain:
    """Main blockchain implementation for audit trails"""
    
    def __init__(self, db_path: str = "blockchain_audit.db"):
        self.db_path = Path(db_path)
        self.blocks: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.crypto_manager = CryptographicManager()
        self.validator_id = self._generate_validator_id()
        
        # Threading
        self.lock = threading.RLock()
        self.mining_active = False
        self.mining_thread = None
        
        # Configuration
        self.max_transactions_per_block = 10
        self.mining_difficulty = 4
        self.auto_mining_interval = 300  # 5 minutes
        
        # Initialize
        self.init_database()
        self.load_blockchain()
        
        # Generate key pair if not exists
        if not self.crypto_manager.key_pair_generated:
            self.crypto_manager.generate_key_pair()
        
        # Create genesis block if blockchain is empty
        if not self.blocks:
            self.create_genesis_block()
    
    def _generate_validator_id(self) -> str:
        """Generate unique validator ID"""
        return f"validator_{secrets.token_hex(8)}"
    
    def init_database(self):
        """Initialize SQLite database for blockchain storage"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Blocks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blocks (
                block_id INTEGER PRIMARY KEY,
                previous_hash TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                transactions TEXT NOT NULL,
                merkle_root TEXT NOT NULL,
                nonce INTEGER NOT NULL,
                block_hash TEXT NOT NULL,
                validator TEXT NOT NULL,
                difficulty INTEGER NOT NULL
            )
        ''')
        
        # Transactions table for quick searching
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                block_id INTEGER,
                transaction_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                data_hash TEXT NOT NULL,
                metadata TEXT NOT NULL,
                digital_signature TEXT,
                compliance_flags TEXT,
                risk_level TEXT DEFAULT 'low',
                FOREIGN KEY (block_id) REFERENCES blocks (block_id)
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(transaction_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp)')
        
        conn.commit()
        conn.close()
    
    def create_genesis_block(self):
        """Create the first block in the blockchain"""
        genesis_transaction = Transaction(
            transaction_id="genesis_tx",
            transaction_type=TransactionType.SYSTEM_EVENT,
            timestamp=datetime.now(),
            user_id="system",
            action="blockchain_initialized",
            data_hash=self.crypto_manager.hash_data("genesis"),
            metadata={"description": "Genesis block for LLM Risk Visualizer audit blockchain"}
        )
        
        genesis_block = Block(
            block_id=0,
            previous_hash="0" * 64,
            timestamp=datetime.now(),
            transactions=[genesis_transaction],
            merkle_root="",
            nonce=0,
            block_hash="",
            validator=self.validator_id
        )
        
        genesis_block.mine_block(self.mining_difficulty)
        
        with self.lock:
            self.blocks.append(genesis_block)
            self.save_block_to_db(genesis_block)
        
        logger.info("Genesis block created")
    
    def add_transaction(self, transaction_type: TransactionType, user_id: str, 
                       action: str, data: Any, metadata: Dict[str, Any] = None,
                       risk_level: str = "low", compliance_flags: List[str] = None) -> str:
        """Add new transaction to pending transactions"""
        
        transaction_id = f"tx_{int(time.time() * 1000)}_{secrets.token_hex(4)}"
        data_hash = self.crypto_manager.hash_data(data)
        
        transaction = Transaction(
            transaction_id=transaction_id,
            transaction_type=transaction_type,
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            data_hash=data_hash,
            metadata=metadata or {},
            compliance_flags=compliance_flags or [],
            risk_level=risk_level
        )
        
        # Sign transaction
        transaction_data = json.dumps(transaction.to_dict(), sort_keys=True)
        signature = self.crypto_manager.sign_data(transaction_data)
        transaction.digital_signature = signature
        
        with self.lock:
            self.pending_transactions.append(transaction)
        
        logger.info(f"Transaction {transaction_id} added to pending pool")
        
        # Trigger mining if enough transactions
        if len(self.pending_transactions) >= self.max_transactions_per_block:
            self.mine_pending_block()
        
        return transaction_id
    
    def mine_pending_block(self) -> Optional[Block]:
        """Mine a new block with pending transactions"""
        if not self.pending_transactions:
            return None
        
        with self.lock:
            # Get transactions for new block
            transactions_to_mine = self.pending_transactions[:self.max_transactions_per_block]
            self.pending_transactions = self.pending_transactions[self.max_transactions_per_block:]
            
            # Create new block
            previous_hash = self.blocks[-1].block_hash if self.blocks else "0" * 64
            new_block_id = len(self.blocks)
            
            new_block = Block(
                block_id=new_block_id,
                previous_hash=previous_hash,
                timestamp=datetime.now(),
                transactions=transactions_to_mine,
                merkle_root="",
                nonce=0,
                block_hash="",
                validator=self.validator_id,
                difficulty=self.mining_difficulty
            )
            
            # Mine the block
            logger.info(f"Mining block {new_block_id} with {len(transactions_to_mine)} transactions...")
            mining_success = new_block.mine_block(self.mining_difficulty)
            
            if mining_success:
                self.blocks.append(new_block)
                self.save_block_to_db(new_block)
                
                logger.info(f"Block {new_block_id} successfully mined and added to blockchain")
                return new_block
            else:
                # Put transactions back in pending pool
                self.pending_transactions = transactions_to_mine + self.pending_transactions
                logger.error(f"Failed to mine block {new_block_id}")
                return None
    
    def save_block_to_db(self, block: Block):
        """Save block to database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Save block
            cursor.execute('''
                INSERT OR REPLACE INTO blocks 
                (block_id, previous_hash, timestamp, transactions, merkle_root, 
                 nonce, block_hash, validator, difficulty)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                block.block_id,
                block.previous_hash,
                block.timestamp.isoformat(),
                json.dumps([tx.to_dict() for tx in block.transactions]),
                block.merkle_root,
                block.nonce,
                block.block_hash,
                block.validator,
                block.difficulty
            ))
            
            # Save individual transactions
            for transaction in block.transactions:
                cursor.execute('''
                    INSERT OR REPLACE INTO transactions
                    (transaction_id, block_id, transaction_type, timestamp, user_id, 
                     action, data_hash, metadata, digital_signature, compliance_flags, risk_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    transaction.transaction_id,
                    block.block_id,
                    transaction.transaction_type.value,
                    transaction.timestamp.isoformat(),
                    transaction.user_id,
                    transaction.action,
                    transaction.data_hash,
                    json.dumps(transaction.metadata),
                    transaction.digital_signature,
                    json.dumps(transaction.compliance_flags),
                    transaction.risk_level
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database save error: {e}")
    
    def load_blockchain(self):
        """Load blockchain from database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM blocks ORDER BY block_id')
            block_rows = cursor.fetchall()
            
            for row in block_rows:
                (block_id, previous_hash, timestamp, transactions_json, 
                 merkle_root, nonce, block_hash, validator, difficulty) = row
                
                # Parse transactions
                transactions_data = json.loads(transactions_json)
                transactions = [Transaction.from_dict(tx_data) for tx_data in transactions_data]
                
                block = Block(
                    block_id=block_id,
                    previous_hash=previous_hash,
                    timestamp=datetime.fromisoformat(timestamp),
                    transactions=transactions,
                    merkle_root=merkle_root,
                    nonce=nonce,
                    block_hash=block_hash,
                    validator=validator,
                    difficulty=difficulty
                )
                
                self.blocks.append(block)
            
            conn.close()
            
            if self.blocks:
                logger.info(f"Loaded {len(self.blocks)} blocks from database")
            
        except Exception as e:
            logger.error(f"Blockchain loading error: {e}")
    
    def validate_blockchain(self) -> Tuple[bool, List[str]]:
        """Validate entire blockchain integrity"""
        issues = []
        
        if not self.blocks:
            return True, []
        
        try:
            # Validate genesis block
            if self.blocks[0].block_id != 0:
                issues.append("Genesis block ID is not 0")
            
            if self.blocks[0].previous_hash != "0" * 64:
                issues.append("Genesis block previous hash is invalid")
            
            # Validate each block
            for i, block in enumerate(self.blocks):
                validation_status = block.validate(
                    self.blocks[i-1].block_hash if i > 0 else None
                )
                
                if validation_status != BlockValidationStatus.VALID:
                    issues.append(f"Block {i} validation failed: {validation_status.value}")
                
                # Check block sequence
                if block.block_id != i:
                    issues.append(f"Block {i} has incorrect block_id: {block.block_id}")
            
            # Check hash chain
            for i in range(1, len(self.blocks)):
                if self.blocks[i].previous_hash != self.blocks[i-1].block_hash:
                    issues.append(f"Block {i} previous_hash doesn't match block {i-1} hash")
            
            is_valid = len(issues) == 0
            
            if is_valid:
                logger.info("Blockchain validation successful")
            else:
                logger.warning(f"Blockchain validation failed with {len(issues)} issues")
            
            return is_valid, issues
            
        except Exception as e:
            logger.error(f"Blockchain validation error: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    def get_transaction_history(self, user_id: str = None, 
                              transaction_type: TransactionType = None,
                              start_date: datetime = None,
                              end_date: datetime = None,
                              limit: int = 100) -> List[Transaction]:
        """Get transaction history with filters"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            query = "SELECT * FROM transactions WHERE 1=1"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if transaction_type:
                query += " AND transaction_type = ?"
                params.append(transaction_type.value)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            transactions = []
            for row in rows:
                (tx_id, block_id, tx_type, timestamp, user_id, action, 
                 data_hash, metadata, signature, compliance_flags, risk_level) = row
                
                transaction = Transaction(
                    transaction_id=tx_id,
                    transaction_type=TransactionType(tx_type),
                    timestamp=datetime.fromisoformat(timestamp),
                    user_id=user_id,
                    action=action,
                    data_hash=data_hash,
                    metadata=json.loads(metadata),
                    digital_signature=signature,
                    compliance_flags=json.loads(compliance_flags) if compliance_flags else [],
                    risk_level=risk_level
                )
                
                transactions.append(transaction)
            
            conn.close()
            return transactions
            
        except Exception as e:
            logger.error(f"Transaction history query error: {e}")
            return []
    
    def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        stats = {
            'total_blocks': len(self.blocks),
            'total_transactions': sum(len(block.transactions) for block in self.blocks),
            'pending_transactions': len(self.pending_transactions),
            'blockchain_size_mb': 0,
            'average_block_time': 0,
            'transaction_types': {},
            'validator_stats': {},
            'compliance_stats': {},
            'risk_level_stats': {}
        }
        
        if not self.blocks:
            return stats
        
        try:
            # Calculate blockchain size
            if self.db_path.exists():
                stats['blockchain_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
            
            # Calculate average block time
            if len(self.blocks) > 1:
                total_time = (self.blocks[-1].timestamp - self.blocks[0].timestamp).total_seconds()
                stats['average_block_time'] = total_time / (len(self.blocks) - 1)
            
            # Analyze transactions
            all_transactions = []
            for block in self.blocks:
                all_transactions.extend(block.transactions)
            
            # Transaction type distribution
            type_counts = {}
            validator_counts = {}
            compliance_counts = {}
            risk_counts = {}
            
            for transaction in all_transactions:
                # Transaction types
                tx_type = transaction.transaction_type.value
                type_counts[tx_type] = type_counts.get(tx_type, 0) + 1
                
                # Risk levels
                risk_level = transaction.risk_level
                risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
                
                # Compliance flags
                for flag in transaction.compliance_flags:
                    compliance_counts[flag] = compliance_counts.get(flag, 0) + 1
            
            # Validator statistics
            for block in self.blocks:
                validator = block.validator
                validator_counts[validator] = validator_counts.get(validator, 0) + 1
            
            stats['transaction_types'] = type_counts
            stats['validator_stats'] = validator_counts
            stats['compliance_stats'] = compliance_counts
            stats['risk_level_stats'] = risk_counts
            
        except Exception as e:
            logger.error(f"Stats calculation error: {e}")
        
        return stats
    
    def start_auto_mining(self):
        """Start automatic mining of pending transactions"""
        if self.mining_active:
            return
        
        self.mining_active = True
        
        def mining_loop():
            while self.mining_active:
                try:
                    if self.pending_transactions:
                        logger.info("Auto-mining pending transactions...")
                        self.mine_pending_block()
                    
                    time.sleep(self.auto_mining_interval)
                    
                except Exception as e:
                    logger.error(f"Auto-mining error: {e}")
                    time.sleep(60)
        
        self.mining_thread = threading.Thread(target=mining_loop, daemon=True)
        self.mining_thread.start()
        
        logger.info("Auto-mining started")
    
    def stop_auto_mining(self):
        """Stop automatic mining"""
        self.mining_active = False
        if self.mining_thread:
            self.mining_thread.join(timeout=10)
        
        logger.info("Auto-mining stopped")

# Compliance and audit helper functions

def log_risk_assessment(blockchain: AuditBlockchain, user_id: str, 
                       assessment_data: Dict[str, Any], risk_score: float):
    """Log risk assessment to blockchain"""
    risk_level = "critical" if risk_score > 0.8 else "high" if risk_score > 0.6 else "medium" if risk_score > 0.3 else "low"
    
    compliance_flags = []
    if risk_score > 0.7:
        compliance_flags.append("high_risk_alert")
    if "bias" in assessment_data:
        compliance_flags.append("bias_check")
    
    return blockchain.add_transaction(
        transaction_type=TransactionType.RISK_ASSESSMENT,
        user_id=user_id,
        action="risk_assessment_performed",
        data=assessment_data,
        metadata={
            "risk_score": risk_score,
            "assessment_timestamp": datetime.now().isoformat(),
            "model_version": assessment_data.get("model_version", "unknown")
        },
        risk_level=risk_level,
        compliance_flags=compliance_flags
    )

def log_user_action(blockchain: AuditBlockchain, user_id: str, action: str, 
                   target: str, details: Dict[str, Any] = None):
    """Log user action to blockchain"""
    return blockchain.add_transaction(
        transaction_type=TransactionType.USER_ACTION,
        user_id=user_id,
        action=action,
        data={"target": target, "details": details or {}},
        metadata={
            "action_timestamp": datetime.now().isoformat(),
            "target": target
        }
    )

def log_data_access(blockchain: AuditBlockchain, user_id: str, 
                   data_source: str, access_type: str, data_hash: str):
    """Log data access to blockchain"""
    compliance_flags = ["data_access"]
    if "sensitive" in data_source.lower():
        compliance_flags.append("sensitive_data_access")
    
    return blockchain.add_transaction(
        transaction_type=TransactionType.DATA_ACCESS,
        user_id=user_id,
        action=f"data_{access_type}",
        data={"source": data_source, "hash": data_hash},
        metadata={
            "access_timestamp": datetime.now().isoformat(),
            "data_source": data_source,
            "access_type": access_type
        },
        compliance_flags=compliance_flags
    )

def log_compliance_check(blockchain: AuditBlockchain, user_id: str, 
                        compliance_standard: str, result: str, details: Dict[str, Any]):
    """Log compliance check to blockchain"""
    risk_level = "high" if result == "failed" else "low"
    
    return blockchain.add_transaction(
        transaction_type=TransactionType.COMPLIANCE_CHECK,
        user_id=user_id,
        action=f"compliance_check_{result}",
        data={"standard": compliance_standard, "result": result, "details": details},
        metadata={
            "check_timestamp": datetime.now().isoformat(),
            "compliance_standard": compliance_standard,
            "result": result
        },
        risk_level=risk_level,
        compliance_flags=[compliance_standard, f"compliance_{result}"]
    )

# Streamlit Integration Functions

def initialize_blockchain_system():
    """Initialize blockchain audit system"""
    if 'audit_blockchain' not in st.session_state:
        st.session_state.audit_blockchain = AuditBlockchain()
        st.session_state.audit_blockchain.start_auto_mining()
    
    return st.session_state.audit_blockchain

def render_blockchain_dashboard():
    """Render blockchain audit dashboard"""
    st.header("🔗 Blockchain Audit Trail")
    
    blockchain = initialize_blockchain_system()
    
    # Blockchain overview
    stats = blockchain.get_blockchain_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Blocks", stats['total_blocks'])
    
    with col2:
        st.metric("Total Transactions", stats['total_transactions'])
    
    with col3:
        st.metric("Pending Transactions", stats['pending_transactions'])
    
    with col4:
        st.metric("Blockchain Size", f"{stats['blockchain_size_mb']:.2f} MB")
    
    # Blockchain validation status
    is_valid, issues = blockchain.validate_blockchain()
    
    if is_valid:
        st.success("✅ Blockchain integrity verified")
    else:
        st.error(f"❌ Blockchain validation failed ({len(issues)} issues)")
        with st.expander("View Issues"):
            for issue in issues:
                st.write(f"• {issue}")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Analytics", 
        "🔍 Transaction History", 
        "⛏️ Mining", 
        "📋 Audit Reports",
        "⚙️ Settings"
    ])
    
    with tab1:
        st.subheader("Blockchain Analytics")
        
        # Transaction type distribution
        if stats['transaction_types']:
            type_data = []
            for tx_type, count in stats['transaction_types'].items():
                type_data.append({
                    'Transaction Type': tx_type.replace('_', ' ').title(),
                    'Count': count
                })
            
            type_df = pd.DataFrame(type_data)
            
            fig_types = px.pie(type_df, values='Count', names='Transaction Type',
                             title='Transaction Types Distribution')
            st.plotly_chart(fig_types, use_container_width=True)
        
        # Risk level distribution
        if stats['risk_level_stats']:
            risk_data = []
            for risk_level, count in stats['risk_level_stats'].items():
                risk_data.append({
                    'Risk Level': risk_level.title(),
                    'Count': count
                })
            
            risk_df = pd.DataFrame(risk_data)
            
            fig_risk = px.bar(risk_df, x='Risk Level', y='Count',
                            title='Risk Level Distribution',
                            color='Risk Level',
                            color_discrete_map={
                                'Critical': '#ff0000',
                                'High': '#ff4444',
                                'Medium': '#ff9500',
                                'Low': '#36a64f'
                            })
            st.plotly_chart(fig_risk, use_container_width=True)
        
        # Compliance statistics
        if stats['compliance_stats']:
            st.subheader("Compliance Events")
            
            compliance_data = []
            for flag, count in stats['compliance_stats'].items():
                compliance_data.append({
                    'Compliance Flag': flag.replace('_', ' ').title(),
                    'Count': count
                })
            
            compliance_df = pd.DataFrame(compliance_data)
            st.dataframe(compliance_df, use_container_width=True)
    
    with tab2:
        st.subheader("Transaction History")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            user_filter = st.text_input("Filter by User ID")
        
        with col2:
            type_filter = st.selectbox(
                "Filter by Transaction Type",
                ["All"] + [t.value for t in TransactionType]
            )
        
        with col3:
            days_back = st.slider("Days Back", 1, 30, 7)
        
        # Get filtered transactions
        start_date = datetime.now() - timedelta(days=days_back)
        tx_type = None if type_filter == "All" else TransactionType(type_filter)
        
        transactions = blockchain.get_transaction_history(
            user_id=user_filter if user_filter else None,
            transaction_type=tx_type,
            start_date=start_date,
            limit=100
        )
        
        if transactions:
            st.write(f"Found {len(transactions)} transactions:")
            
            # Create transaction display
            for i, tx in enumerate(transactions[:20]):  # Show first 20
                risk_colors = {
                    'critical': '🔴',
                    'high': '🟠', 
                    'medium': '🟡',
                    'low': '🟢'
                }
                
                risk_icon = risk_colors.get(tx.risk_level, '⚪')
                
                with st.expander(f"{risk_icon} {tx.action} - {tx.user_id} ({tx.timestamp.strftime('%Y-%m-%d %H:%M:%S')})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Transaction ID:** {tx.transaction_id}")
                        st.write(f"**Type:** {tx.transaction_type.value}")
                        st.write(f"**User:** {tx.user_id}")
                        st.write(f"**Action:** {tx.action}")
                        st.write(f"**Risk Level:** {tx.risk_level}")
                    
                    with col2:
                        st.write(f"**Data Hash:** {tx.data_hash[:16]}...")
                        st.write(f"**Timestamp:** {tx.timestamp}")
                        if tx.compliance_flags:
                            st.write(f"**Compliance Flags:** {', '.join(tx.compliance_flags)}")
                        if tx.digital_signature:
                            st.write("**Digitally Signed:** ✅")
                    
                    if tx.metadata:
                        with st.expander("Metadata"):
                            st.json(tx.metadata)
            
            if len(transactions) > 20:
                st.info(f"Showing 20 of {len(transactions)} transactions")
        else:
            st.info("No transactions found matching the filters")
    
    with tab3:
        st.subheader("Blockchain Mining")
        
        # Mining status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mining_status = "🟢 Active" if blockchain.mining_active else "🔴 Inactive"
            st.write(f"**Auto-Mining Status:** {mining_status}")
        
        with col2:
            st.write(f"**Mining Difficulty:** {blockchain.mining_difficulty}")
        
        with col3:
            st.write(f"**Pending Transactions:** {len(blockchain.pending_transactions)}")
        
        # Mining controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⛏️ Mine Pending Block"):
                if blockchain.pending_transactions:
                    with st.spinner("Mining block..."):
                        new_block = blockchain.mine_pending_block()
                        if new_block:
                            st.success(f"Block {new_block.block_id} mined successfully!")
                        else:
                            st.error("Mining failed")
                    st.rerun()
                else:
                    st.warning("No pending transactions to mine")
        
        with col2:
            if blockchain.mining_active:
                if st.button("⏹️ Stop Auto-Mining"):
                    blockchain.stop_auto_mining()
                    st.success("Auto-mining stopped")
                    st.rerun()
            else:
                if st.button("▶️ Start Auto-Mining"):
                    blockchain.start_auto_mining()
                    st.success("Auto-mining started")
                    st.rerun()
        
        with col3:
            if st.button("🔄 Validate Blockchain"):
                with st.spinner("Validating blockchain..."):
                    is_valid, issues = blockchain.validate_blockchain()
                    if is_valid:
                        st.success("Blockchain is valid!")
                    else:
                        st.error(f"Validation failed: {len(issues)} issues found")
        
        # Recent blocks
        if blockchain.blocks:
            st.subheader("Recent Blocks")
            
            recent_blocks = blockchain.blocks[-5:]  # Last 5 blocks
            
            for block in reversed(recent_blocks):
                with st.expander(f"Block {block.block_id} - {len(block.transactions)} transactions"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Block Hash:** {block.block_hash[:16]}...")
                        st.write(f"**Previous Hash:** {block.previous_hash[:16]}...")
                        st.write(f"**Merkle Root:** {block.merkle_root[:16]}...")
                    
                    with col2:
                        st.write(f"**Timestamp:** {block.timestamp}")
                        st.write(f"**Nonce:** {block.nonce}")
                        st.write(f"**Validator:** {block.validator}")
    
    with tab4:
        st.subheader("Audit Reports")
        
        # Generate compliance report
        if st.button("📋 Generate Compliance Report"):
            with st.spinner("Generating compliance report..."):
                # Get recent transactions for report
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                all_transactions = blockchain.get_transaction_history(
                    start_date=start_date,
                    limit=1000
                )
                
                if all_transactions:
                    # Compliance summary
                    st.write("**30-Day Compliance Summary:**")
                    
                    compliance_summary = {}
                    risk_summary = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
                    user_activity = {}
                    
                    for tx in all_transactions:
                        # Compliance flags
                        for flag in tx.compliance_flags:
                            compliance_summary[flag] = compliance_summary.get(flag, 0) + 1
                        
                        # Risk levels
                        if tx.risk_level in risk_summary:
                            risk_summary[tx.risk_level] += 1
                        
                        # User activity
                        user_activity[tx.user_id] = user_activity.get(tx.user_id, 0) + 1
                    
                    # Display summaries
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Risk Level Summary:**")
                        for level, count in risk_summary.items():
                            if count > 0:
                                st.write(f"• {level.title()}: {count}")
                    
                    with col2:
                        st.write("**Top Compliance Events:**")
                        top_compliance = sorted(compliance_summary.items(), 
                                              key=lambda x: x[1], reverse=True)[:5]
                        for flag, count in top_compliance:
                            st.write(f"• {flag.replace('_', ' ').title()}: {count}")
                    
                    # User activity report
                    st.write("**Most Active Users:**")
                    top_users = sorted(user_activity.items(), 
                                     key=lambda x: x[1], reverse=True)[:10]
                    
                    user_data = []
                    for user, count in top_users:
                        user_data.append({
                            'User ID': user,
                            'Transaction Count': count
                        })
                    
                    user_df = pd.DataFrame(user_data)
                    st.dataframe(user_df, use_container_width=True)
                else:
                    st.info("No transactions found in the last 30 days")
        
        # Export options
        st.subheader("Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Transaction History"):
                transactions = blockchain.get_transaction_history(limit=1000)
                if transactions:
                    export_data = []
                    for tx in transactions:
                        export_data.append({
                            'Transaction ID': tx.transaction_id,
                            'Type': tx.transaction_type.value,
                            'Timestamp': tx.timestamp.isoformat(),
                            'User ID': tx.user_id,
                            'Action': tx.action,
                            'Risk Level': tx.risk_level,
                            'Data Hash': tx.data_hash,
                            'Compliance Flags': ', '.join(tx.compliance_flags)
                        })
                    
                    export_df = pd.DataFrame(export_data)
                    csv = export_df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"blockchain_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv'
                    )
                else:
                    st.warning("No transactions to export")
        
        with col2:
            if st.button("📊 Export Blockchain Stats"):
                stats_json = json.dumps(stats, indent=2, default=str)
                
                st.download_button(
                    label="Download JSON",
                    data=stats_json,
                    file_name=f"blockchain_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime='application/json'
                )
    
    with tab5:
        st.subheader("Blockchain Settings")
        
        # Mining configuration
        st.write("**Mining Configuration:**")
        
        new_difficulty = st.slider("Mining Difficulty", 1, 8, blockchain.mining_difficulty)
        new_max_tx = st.slider("Max Transactions per Block", 5, 50, blockchain.max_transactions_per_block)
        new_mining_interval = st.slider("Auto-mining Interval (minutes)", 1, 60, blockchain.auto_mining_interval // 60)
        
        if st.button("Update Mining Settings"):
            blockchain.mining_difficulty = new_difficulty
            blockchain.max_transactions_per_block = new_max_tx
            blockchain.auto_mining_interval = new_mining_interval * 60
            st.success("Mining settings updated!")
        
        # Blockchain maintenance
        st.write("**Maintenance Operations:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Clear Pending Transactions"):
                blockchain.pending_transactions.clear()
                st.success("Pending transactions cleared!")
                st.rerun()
        
        with col2:
            if st.button("💾 Force Database Sync"):
                # Re-save all blocks to ensure database consistency
                for block in blockchain.blocks:
                    blockchain.save_block_to_db(block)
                st.success("Database synchronized!")
        
        # Cryptographic information
        st.write("**Cryptographic Information:**")
        
        st.write(f"**Validator ID:** {blockchain.validator_id}")
        st.write(f"**Key Pair Generated:** {'✅' if blockchain.crypto_manager.key_pair_generated else '❌'}")
        
        if st.button("🔑 Regenerate Key Pair"):
            private_key, public_key = blockchain.crypto_manager.generate_key_pair()
            if private_key:
                st.success("New key pair generated!")
                with st.expander("Public Key (Safe to Share)"):
                    st.text_area("Public Key", public_key, height=200)
            else:
                st.error("Key generation failed!")

if __name__ == "__main__":
    # Example usage and testing
    
    # Initialize blockchain
    blockchain = AuditBlockchain()
    
    print("Testing blockchain audit system...")
    
    # Add some sample transactions
    tx1_id = log_risk_assessment(
        blockchain, 
        "user123", 
        {"model": "GPT-4", "risk_score": 0.75}, 
        0.75
    )
    
    tx2_id = log_user_action(
        blockchain,
        "user456",
        "view_dashboard",
        "risk_dashboard",
        {"page": "main", "filters": ["high_risk"]}
    )
    
    tx3_id = log_data_access(
        blockchain,
        "user789",
        "sensitive_customer_data",
        "read",
        "abc123def456"
    )
    
    tx4_id = log_compliance_check(
        blockchain,
        "system",
        "GDPR",
        "passed",
        {"checks": ["data_retention", "consent_tracking"]}
    )
    
    print(f"Added transactions: {tx1_id}, {tx2_id}, {tx3_id}, {tx4_id}")
    
    # Mine pending transactions
    new_block = blockchain.mine_pending_block()
    if new_block:
        print(f"Mined block {new_block.block_id} with {len(new_block.transactions)} transactions")
    
    # Validate blockchain
    is_valid, issues = blockchain.validate_blockchain()
    print(f"Blockchain valid: {is_valid}")
    if issues:
        print(f"Issues: {issues}")
    
    # Get statistics
    stats = blockchain.get_blockchain_stats()
    print(f"Blockchain stats: {stats['total_blocks']} blocks, {stats['total_transactions']} transactions")
    
    # Get transaction history
    history = blockchain.get_transaction_history(limit=5)
    print(f"Recent transactions: {len(history)}")
    
    for tx in history:
        print(f"  - {tx.transaction_id}: {tx.action} by {tx.user_id}")
    
    print("Blockchain audit system test completed!")