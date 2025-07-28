"""
Quantum Computing Preparation and Post-Quantum Cryptography Module
Implements quantum-safe security measures and prepares for quantum computing integration
"""

import json
import time
import hashlib
import hmac
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import defaultdict, deque
import uuid
import base64
import threading
from concurrent.futures import ThreadPoolExecutor

# Cryptographic libraries
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("Cryptography library not available. Some features will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantumThreatLevel(Enum):
    """Quantum threat assessment levels"""
    MINIMAL = "minimal"          # No quantum threat expected
    LOW = "low"                 # Quantum threat in 15+ years
    MODERATE = "moderate"       # Quantum threat in 10-15 years
    HIGH = "high"              # Quantum threat in 5-10 years
    CRITICAL = "critical"      # Quantum threat imminent

class CryptographicAlgorithm(Enum):
    """Cryptographic algorithm types"""
    RSA = "rsa"
    ECC = "ecc"
    AES = "aes"
    SHA256 = "sha256"
    KYBER = "kyber"            # Post-quantum KEM
    DILITHIUM = "dilithium"    # Post-quantum signatures
    SPHINCS = "sphincs"        # Post-quantum signatures
    LATTICE_BASED = "lattice_based"
    HASH_BASED = "hash_based"
    CODE_BASED = "code_based"

class QuantumResistance(Enum):
    """Quantum resistance levels"""
    VULNERABLE = "vulnerable"         # Broken by quantum computers
    PARTIALLY_RESISTANT = "partial"  # Some resistance but not proven
    QUANTUM_SAFE = "quantum_safe"    # Believed to be quantum-resistant
    QUANTUM_PROVEN = "quantum_proven" # Mathematically proven quantum-safe

@dataclass
class CryptographicAsset:
    """Cryptographic asset that needs quantum assessment"""
    asset_id: str
    name: str
    algorithm: CryptographicAlgorithm
    key_size: int
    usage_context: str
    
    # Security properties
    current_security_level: int  # in bits
    quantum_security_level: int  # security against quantum attacks
    quantum_resistance: QuantumResistance
    
    # Lifecycle information
    created_date: datetime
    expiry_date: Optional[datetime] = None
    last_assessment: Optional[datetime] = None
    
    # Migration status
    migration_priority: str = "medium"  # low, medium, high, critical
    migration_planned: bool = False
    replacement_algorithm: Optional[CryptographicAlgorithm] = None
    
    # Usage statistics
    usage_frequency: float = 0.0
    performance_impact: float = 0.0
    migration_cost_estimate: float = 0.0

@dataclass
class QuantumThreatAssessment:
    """Quantum threat assessment for the system"""
    assessment_id: str
    timestamp: datetime
    
    # Threat timeline
    cryptographically_relevant_quantum_computer_eta: int  # years
    threat_level: QuantumThreatLevel
    confidence_level: float  # 0.0 to 1.0
    
    # Impact analysis
    vulnerable_assets_count: int
    critical_vulnerabilities: List[str]
    estimated_migration_time: int  # months
    estimated_migration_cost: float
    
    # Recommendations
    immediate_actions: List[str]
    migration_roadmap: Dict[str, Any]
    risk_mitigation_strategies: List[str]

@dataclass
class PostQuantumMigrationPlan:
    """Migration plan for post-quantum cryptography"""
    plan_id: str
    created_date: datetime
    target_completion_date: datetime
    
    # Migration phases
    phases: List[Dict[str, Any]]
    current_phase: int = 0
    
    # Assets to migrate
    assets_to_migrate: List[str]
    migration_priorities: Dict[str, str]
    
    # Resources and timeline
    estimated_duration_months: int
    estimated_cost: float
    required_resources: List[str]
    
    # Progress tracking
    completion_percentage: float = 0.0
    milestones_completed: List[str] = None
    
    # Risk assessment
    migration_risks: List[str] = None
    risk_mitigation_measures: List[str] = None
    
    def __post_init__(self):
        if self.milestones_completed is None:
            self.milestones_completed = []
        if self.migration_risks is None:
            self.migration_risks = []
        if self.risk_mitigation_measures is None:
            self.risk_mitigation_measures = []

class PostQuantumCryptographyEngine:
    """Post-quantum cryptography implementation engine"""
    
    def __init__(self):
        self.supported_algorithms = {
            CryptographicAlgorithm.KYBER: {
                "type": "key_encapsulation",
                "security_level": 256,
                "key_size": 1568,  # bytes for public key
                "performance_factor": 0.8
            },
            CryptographicAlgorithm.DILITHIUM: {
                "type": "digital_signature", 
                "security_level": 256,
                "signature_size": 2420,  # bytes
                "performance_factor": 0.6
            },
            CryptographicAlgorithm.SPHINCS: {
                "type": "digital_signature",
                "security_level": 256,
                "signature_size": 17088,  # bytes
                "performance_factor": 0.3
            },
            CryptographicAlgorithm.LATTICE_BASED: {
                "type": "general_purpose",
                "security_level": 128,
                "key_size": 1024,
                "performance_factor": 0.7
            }
        }
        
        # Simulated quantum-safe implementations
        self.quantum_safe_keys = {}
        self.hybrid_implementations = {}
        
    def generate_quantum_safe_key(self, algorithm: CryptographicAlgorithm, 
                                 key_id: str) -> Dict[str, Any]:
        """Generate quantum-safe cryptographic key"""
        
        if algorithm not in self.supported_algorithms:
            raise ValueError(f"Algorithm {algorithm.value} not supported")
        
        algo_spec = self.supported_algorithms[algorithm]
        
        # Simulate key generation (in real implementation, would use actual PQC libraries)
        key_data = {
            "key_id": key_id,
            "algorithm": algorithm.value,
            "key_size": algo_spec["key_size"],
            "security_level": algo_spec["security_level"],
            "public_key": base64.b64encode(secrets.token_bytes(algo_spec["key_size"])).decode(),
            "private_key": base64.b64encode(secrets.token_bytes(algo_spec["key_size"] * 2)).decode(),
            "created_date": datetime.now().isoformat(),
            "performance_factor": algo_spec["performance_factor"]
        }
        
        # Store key
        self.quantum_safe_keys[key_id] = key_data
        
        logger.info(f"Generated quantum-safe key: {key_id} using {algorithm.value}")
        return key_data
    
    def create_hybrid_system(self, classical_algorithm: CryptographicAlgorithm,
                           quantum_safe_algorithm: CryptographicAlgorithm,
                           system_id: str) -> Dict[str, Any]:
        """Create hybrid classical + post-quantum cryptographic system"""
        
        hybrid_system = {
            "system_id": system_id,
            "classical_algorithm": classical_algorithm.value,
            "quantum_safe_algorithm": quantum_safe_algorithm.value,
            "created_date": datetime.now().isoformat(),
            "security_level": min(256, self.supported_algorithms.get(quantum_safe_algorithm, {}).get("security_level", 128)),
            "performance_overhead": 1.0 - self.supported_algorithms.get(quantum_safe_algorithm, {}).get("performance_factor", 0.5),
            "migration_ready": True
        }
        
        self.hybrid_implementations[system_id] = hybrid_system
        
        logger.info(f"Created hybrid cryptographic system: {system_id}")
        return hybrid_system
    
    def simulate_quantum_attack(self, target_algorithm: CryptographicAlgorithm,
                               key_size: int) -> Dict[str, Any]:
        """Simulate quantum attack against cryptographic algorithm"""
        
        # Quantum attack simulation based on known quantum algorithms
        quantum_algorithms = {
            CryptographicAlgorithm.RSA: {
                "attack_method": "Shor's Algorithm",
                "quantum_speedup": "exponential",
                "qubits_required": key_size * 2,  # Rough estimate
                "time_complexity": "polynomial"
            },
            CryptographicAlgorithm.ECC: {
                "attack_method": "Shor's Algorithm (Elliptic Curve)",
                "quantum_speedup": "exponential", 
                "qubits_required": key_size * 6,  # Rough estimate
                "time_complexity": "polynomial"
            },
            CryptographicAlgorithm.AES: {
                "attack_method": "Grover's Algorithm",
                "quantum_speedup": "quadratic",
                "qubits_required": key_size,
                "time_complexity": "square_root"
            },
            CryptographicAlgorithm.SHA256: {
                "attack_method": "Grover's Algorithm",
                "quantum_speedup": "quadratic",
                "qubits_required": 256,
                "time_complexity": "square_root"
            }
        }
        
        if target_algorithm in quantum_algorithms:
            attack_spec = quantum_algorithms[target_algorithm]
            
            # Calculate effective security reduction
            if attack_spec["quantum_speedup"] == "exponential":
                effective_security = 0  # Completely broken
                break_time_years = 0.001  # Essentially instant
            else:  # quadratic speedup
                effective_security = key_size // 2  # Half the security bits
                break_time_years = 2 ** (effective_security - 20)  # Rough estimate
            
            return {
                "target_algorithm": target_algorithm.value,
                "attack_method": attack_spec["attack_method"],
                "quantum_speedup": attack_spec["quantum_speedup"],
                "qubits_required": attack_spec["qubits_required"],
                "original_security_bits": key_size,
                "effective_security_bits": effective_security,
                "estimated_break_time_years": break_time_years,
                "vulnerability_level": "critical" if effective_security < 80 else "high" if effective_security < 112 else "moderate"
            }
        else:
            # Assume quantum-resistant
            return {
                "target_algorithm": target_algorithm.value,
                "attack_method": "No known efficient quantum attack",
                "quantum_speedup": "none",
                "vulnerability_level": "minimal",
                "estimated_break_time_years": float('inf')
            }
    
    def benchmark_pqc_performance(self, algorithm: CryptographicAlgorithm,
                                 operations: int = 1000) -> Dict[str, Any]:
        """Benchmark post-quantum cryptographic algorithm performance"""
        
        if algorithm not in self.supported_algorithms:
            return {"error": f"Algorithm {algorithm.value} not supported"}
        
        algo_spec = self.supported_algorithms[algorithm]
        base_time = 0.001  # Base operation time in milliseconds
        
        # Simulate performance characteristics
        performance_factor = algo_spec["performance_factor"]
        
        results = {
            "algorithm": algorithm.value,
            "operations_tested": operations,
            "key_generation_time_ms": base_time * 100 / performance_factor,
            "encryption_time_ms": base_time * 10 / performance_factor,
            "decryption_time_ms": base_time * 15 / performance_factor,
            "signature_time_ms": base_time * 20 / performance_factor,
            "verification_time_ms": base_time * 5 / performance_factor,
            "key_size_bytes": algo_spec["key_size"],
            "security_level_bits": algo_spec["security_level"],
            "performance_overhead": (1.0 - performance_factor) * 100,  # Percentage overhead
            "memory_usage_mb": algo_spec["key_size"] / 1024 / 1024 * 2,  # Rough estimate
            "throughput_ops_per_second": operations / (base_time * operations / performance_factor / 1000)
        }
        
        return results

class QuantumThreatAnalyzer:
    """Analyzes quantum threats to cryptographic systems"""
    
    def __init__(self):
        self.cryptographic_assets: Dict[str, CryptographicAsset] = {}
        self.threat_assessments: List[QuantumThreatAssessment] = []
        self.quantum_timeline_estimates = {
            "conservative": 20,  # years
            "moderate": 15,
            "aggressive": 10,
            "breakthrough": 5
        }
        
    def register_cryptographic_asset(self, asset: CryptographicAsset):
        """Register cryptographic asset for quantum threat analysis"""
        self.cryptographic_assets[asset.asset_id] = asset
        logger.info(f"Registered cryptographic asset: {asset.name}")
        
    def assess_quantum_vulnerability(self, asset_id: str) -> Dict[str, Any]:
        """Assess quantum vulnerability of a cryptographic asset"""
        
        if asset_id not in self.cryptographic_assets:
            return {"error": "Asset not found"}
        
        asset = self.cryptographic_assets[asset_id]
        
        # Determine quantum resistance based on algorithm
        vulnerability_matrix = {
            CryptographicAlgorithm.RSA: {
                "quantum_resistance": QuantumResistance.VULNERABLE,
                "break_probability": 1.0,
                "migration_urgency": "critical"
            },
            CryptographicAlgorithm.ECC: {
                "quantum_resistance": QuantumResistance.VULNERABLE,
                "break_probability": 1.0,
                "migration_urgency": "critical"
            },
            CryptographicAlgorithm.AES: {
                "quantum_resistance": QuantumResistance.PARTIALLY_RESISTANT,
                "break_probability": 0.3 if asset.key_size >= 256 else 0.8,
                "migration_urgency": "medium" if asset.key_size >= 256 else "high"
            },
            CryptographicAlgorithm.SHA256: {
                "quantum_resistance": QuantumResistance.PARTIALLY_RESISTANT,
                "break_probability": 0.2,
                "migration_urgency": "medium"
            },
            CryptographicAlgorithm.KYBER: {
                "quantum_resistance": QuantumResistance.QUANTUM_SAFE,
                "break_probability": 0.0,
                "migration_urgency": "none"
            },
            CryptographicAlgorithm.DILITHIUM: {
                "quantum_resistance": QuantumResistance.QUANTUM_SAFE,
                "break_probability": 0.0,
                "migration_urgency": "none"
            }
        }
        
        vuln_data = vulnerability_matrix.get(asset.algorithm, {
            "quantum_resistance": QuantumResistance.VULNERABLE,
            "break_probability": 0.9,
            "migration_urgency": "high"
        })
        
        # Calculate risk score
        risk_factors = {
            "algorithm_vulnerability": vuln_data["break_probability"] * 40,
            "key_size_factor": max(0, (2048 - asset.key_size) / 2048 * 20) if asset.algorithm in [CryptographicAlgorithm.RSA] else 0,
            "usage_frequency": min(asset.usage_frequency * 20, 20),
            "criticality": {"critical": 20, "high": 15, "medium": 10, "low": 5}.get(asset.migration_priority, 10)
        }
        
        total_risk_score = sum(risk_factors.values())
        
        return {
            "asset_id": asset_id,
            "asset_name": asset.name,
            "algorithm": asset.algorithm.value,
            "quantum_resistance": vuln_data["quantum_resistance"].value,
            "break_probability": vuln_data["break_probability"],
            "migration_urgency": vuln_data["migration_urgency"],
            "risk_score": total_risk_score,
            "risk_level": "critical" if total_risk_score > 70 else "high" if total_risk_score > 50 else "medium" if total_risk_score > 30 else "low",
            "risk_factors": risk_factors,
            "recommended_actions": self._generate_recommendations(asset, vuln_data),
            "assessment_date": datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, asset: CryptographicAsset, 
                                vuln_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations for quantum threat mitigation"""
        
        recommendations = []
        
        if vuln_data["quantum_resistance"] == QuantumResistance.VULNERABLE:
            recommendations.append(f"URGENT: Migrate {asset.name} to post-quantum algorithm")
            recommendations.append("Implement hybrid classical+PQC system for transition period")
            
            if asset.algorithm == CryptographicAlgorithm.RSA:
                recommendations.append("Replace RSA with CRYSTALS-Kyber for key exchange")
                recommendations.append("Replace RSA signatures with CRYSTALS-Dilithium")
            elif asset.algorithm == CryptographicAlgorithm.ECC:
                recommendations.append("Replace ECDH with CRYSTALS-Kyber")
                recommendations.append("Replace ECDSA with CRYSTALS-Dilithium")
                
        elif vuln_data["quantum_resistance"] == QuantumResistance.PARTIALLY_RESISTANT:
            if asset.key_size < 256:
                recommendations.append(f"Increase key size for {asset.name} to at least 256 bits")
            recommendations.append("Plan migration to quantum-safe alternatives")
            recommendations.append("Monitor quantum computing developments")
            
        else:
            recommendations.append("Continue monitoring for algorithm updates")
            recommendations.append("Ensure implementation follows best practices")
        
        # Usage-specific recommendations
        if asset.usage_frequency > 0.8:
            recommendations.append("High-usage asset - prioritize performance optimization")
            
        if asset.migration_priority in ["critical", "high"]:
            recommendations.append("Schedule immediate security review")
            
        return recommendations
    
    def perform_system_wide_assessment(self) -> QuantumThreatAssessment:
        """Perform comprehensive quantum threat assessment"""
        
        assessment_id = str(uuid.uuid4())
        vulnerable_assets = []
        critical_vulnerabilities = []
        total_migration_cost = 0.0
        
        # Analyze all registered assets
        for asset_id, asset in self.cryptographic_assets.items():
            vulnerability = self.assess_quantum_vulnerability(asset_id)
            
            if vulnerability["quantum_resistance"] in ["vulnerable", "partially_resistant"]:
                vulnerable_assets.append(asset_id)
                total_migration_cost += asset.migration_cost_estimate
                
                if vulnerability["risk_level"] == "critical":
                    critical_vulnerabilities.append(f"{asset.name}: {vulnerability['migration_urgency']}")
        
        # Determine overall threat level
        vulnerable_ratio = len(vulnerable_assets) / max(1, len(self.cryptographic_assets))
        critical_ratio = len(critical_vulnerabilities) / max(1, len(self.cryptographic_assets))
        
        if critical_ratio > 0.5:
            threat_level = QuantumThreatLevel.CRITICAL
        elif critical_ratio > 0.2:
            threat_level = QuantumThreatLevel.HIGH
        elif vulnerable_ratio > 0.5:
            threat_level = QuantumThreatLevel.MODERATE
        elif vulnerable_ratio > 0.1:
            threat_level = QuantumThreatLevel.LOW
        else:
            threat_level = QuantumThreatLevel.MINIMAL
        
        # Generate assessment
        assessment = QuantumThreatAssessment(
            assessment_id=assessment_id,
            timestamp=datetime.now(),
            cryptographically_relevant_quantum_computer_eta=self.quantum_timeline_estimates["moderate"],
            threat_level=threat_level,
            confidence_level=0.8,
            vulnerable_assets_count=len(vulnerable_assets),
            critical_vulnerabilities=critical_vulnerabilities,
            estimated_migration_time=len(vulnerable_assets) * 2,  # months
            estimated_migration_cost=total_migration_cost,
            immediate_actions=self._generate_immediate_actions(threat_level, critical_vulnerabilities),
            migration_roadmap=self._generate_migration_roadmap(vulnerable_assets),
            risk_mitigation_strategies=self._generate_risk_mitigation_strategies(threat_level)
        )
        
        self.threat_assessments.append(assessment)
        
        return assessment
    
    def _generate_immediate_actions(self, threat_level: QuantumThreatLevel,
                                  critical_vulnerabilities: List[str]) -> List[str]:
        """Generate immediate action items based on threat level"""
        
        actions = []
        
        if threat_level == QuantumThreatLevel.CRITICAL:
            actions.extend([
                "Freeze deployment of new quantum-vulnerable systems",
                "Activate emergency migration protocols",
                "Conduct immediate security review of critical systems",
                "Notify stakeholders of quantum threat status"
            ])
        elif threat_level == QuantumThreatLevel.HIGH:
            actions.extend([
                "Accelerate post-quantum migration planning",
                "Begin pilot implementations of PQC algorithms",
                "Review and update cryptographic policies"
            ])
        elif threat_level == QuantumThreatLevel.MODERATE:
            actions.extend([
                "Develop comprehensive PQC migration strategy",
                "Begin evaluation of post-quantum algorithms",
                "Update risk management frameworks"
            ])
        else:
            actions.extend([
                "Continue monitoring quantum computing developments",
                "Maintain awareness of PQC standardization efforts"
            ])
        
        # Add specific actions for critical vulnerabilities
        if critical_vulnerabilities:
            actions.append(f"Address {len(critical_vulnerabilities)} critical vulnerabilities immediately")
        
        return actions
    
    def _generate_migration_roadmap(self, vulnerable_assets: List[str]) -> Dict[str, Any]:
        """Generate migration roadmap for vulnerable assets"""
        
        if not vulnerable_assets:
            return {"message": "No migration required"}
        
        # Prioritize assets by risk and impact
        prioritized_assets = []
        for asset_id in vulnerable_assets:
            asset = self.cryptographic_assets[asset_id]
            priority_score = asset.usage_frequency * 50 + {"critical": 40, "high": 30, "medium": 20, "low": 10}.get(asset.migration_priority, 20)
            prioritized_assets.append((asset_id, priority_score))
        
        prioritized_assets.sort(key=lambda x: x[1], reverse=True)
        
        # Create phased migration plan
        phases = []
        assets_per_phase = max(1, len(prioritized_assets) // 4)
        
        for i in range(0, len(prioritized_assets), assets_per_phase):
            phase_assets = prioritized_assets[i:i + assets_per_phase]
            phases.append({
                "phase": len(phases) + 1,
                "assets": [asset_id for asset_id, _ in phase_assets],
                "duration_months": len(phase_assets) * 2,
                "estimated_cost": sum(self.cryptographic_assets[asset_id].migration_cost_estimate for asset_id, _ in phase_assets)
            })
        
        return {
            "total_phases": len(phases),
            "total_duration_months": sum(phase["duration_months"] for phase in phases),
            "total_cost": sum(phase["estimated_cost"] for phase in phases),
            "phases": phases
        }
    
    def _generate_risk_mitigation_strategies(self, threat_level: QuantumThreatLevel) -> List[str]:
        """Generate risk mitigation strategies"""
        
        strategies = [
            "Implement crypto-agility in system design",
            "Establish quantum-safe communication channels",
            "Deploy hybrid classical+post-quantum systems",
            "Create incident response plans for quantum breakthroughs",
            "Maintain inventory of all cryptographic assets"
        ]
        
        if threat_level in [QuantumThreatLevel.HIGH, QuantumThreatLevel.CRITICAL]:
            strategies.extend([
                "Implement additional authentication factors",
                "Increase monitoring of cryptographic operations",
                "Prepare for emergency algorithm updates",
                "Consider quantum key distribution for critical communications"
            ])
        
        return strategies

class QuantumReadinessManager:
    """Main quantum readiness and PQC migration manager"""
    
    def __init__(self, db_path: str = "quantum_readiness.db"):
        self.db_path = Path(db_path)
        self.pqc_engine = PostQuantumCryptographyEngine()
        self.threat_analyzer = QuantumThreatAnalyzer()
        self.migration_plans: Dict[str, PostQuantumMigrationPlan] = {}
        
        # Initialize database
        self.init_database()
        
        # Load existing data
        self.load_cryptographic_assets()
        
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Cryptographic assets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS crypto_assets (
                asset_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                key_size INTEGER NOT NULL,
                usage_context TEXT,
                current_security_level INTEGER,
                quantum_security_level INTEGER,
                quantum_resistance TEXT,
                created_date TEXT NOT NULL,
                expiry_date TEXT,
                last_assessment TEXT,
                migration_priority TEXT DEFAULT 'medium',
                migration_planned BOOLEAN DEFAULT 0,
                replacement_algorithm TEXT,
                usage_frequency REAL DEFAULT 0.0,
                performance_impact REAL DEFAULT 0.0,
                migration_cost_estimate REAL DEFAULT 0.0
            )
        ''')
        
        # Quantum assessments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quantum_assessments (
                assessment_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                threat_level TEXT NOT NULL,
                confidence_level REAL NOT NULL,
                vulnerable_assets_count INTEGER,
                critical_vulnerabilities TEXT,
                estimated_migration_time INTEGER,
                estimated_migration_cost REAL,
                immediate_actions TEXT,
                migration_roadmap TEXT,
                risk_mitigation_strategies TEXT
            )
        ''')
        
        # Migration plans table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS migration_plans (
                plan_id TEXT PRIMARY KEY,
                created_date TEXT NOT NULL,
                target_completion_date TEXT NOT NULL,
                phases TEXT NOT NULL,
                current_phase INTEGER DEFAULT 0,
                assets_to_migrate TEXT,
                migration_priorities TEXT,
                estimated_duration_months INTEGER,
                estimated_cost REAL,
                required_resources TEXT,
                completion_percentage REAL DEFAULT 0.0,
                milestones_completed TEXT,
                migration_risks TEXT,
                risk_mitigation_measures TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def load_cryptographic_assets(self):
        """Load cryptographic assets from database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM crypto_assets')
            rows = cursor.fetchall()
            
            for row in rows:
                asset = self._row_to_crypto_asset(row)
                if asset:
                    self.threat_analyzer.register_cryptographic_asset(asset)
            
            conn.close()
            logger.info(f"Loaded {len(rows)} cryptographic assets from database")
            
        except Exception as e:
            logger.error(f"Error loading cryptographic assets: {e}")
    
    def _row_to_crypto_asset(self, row) -> Optional[CryptographicAsset]:
        """Convert database row to CryptographicAsset object"""
        try:
            return CryptographicAsset(
                asset_id=row[0],
                name=row[1],
                algorithm=CryptographicAlgorithm(row[2]),
                key_size=row[3],
                usage_context=row[4] or "",
                current_security_level=row[5],
                quantum_security_level=row[6],
                quantum_resistance=QuantumResistance(row[7]),
                created_date=datetime.fromisoformat(row[8]),
                expiry_date=datetime.fromisoformat(row[9]) if row[9] else None,
                last_assessment=datetime.fromisoformat(row[10]) if row[10] else None,
                migration_priority=row[11] or "medium",
                migration_planned=bool(row[12]),
                replacement_algorithm=CryptographicAlgorithm(row[13]) if row[13] else None,
                usage_frequency=row[14] or 0.0,
                performance_impact=row[15] or 0.0,
                migration_cost_estimate=row[16] or 0.0
            )
        except Exception as e:
            logger.error(f"Error converting row to crypto asset: {e}")
            return None
    
    def register_cryptographic_asset(self, name: str, algorithm: CryptographicAlgorithm,
                                   key_size: int, usage_context: str = "",
                                   migration_priority: str = "medium") -> str:
        """Register new cryptographic asset"""
        
        asset_id = str(uuid.uuid4())
        
        # Determine quantum resistance
        quantum_resistance_map = {
            CryptographicAlgorithm.RSA: QuantumResistance.VULNERABLE,
            CryptographicAlgorithm.ECC: QuantumResistance.VULNERABLE,
            CryptographicAlgorithm.AES: QuantumResistance.PARTIALLY_RESISTANT,
            CryptographicAlgorithm.SHA256: QuantumResistance.PARTIALLY_RESISTANT,
            CryptographicAlgorithm.KYBER: QuantumResistance.QUANTUM_SAFE,
            CryptographicAlgorithm.DILITHIUM: QuantumResistance.QUANTUM_SAFE,
            CryptographicAlgorithm.SPHINCS: QuantumResistance.QUANTUM_SAFE
        }
        
        quantum_resistance = quantum_resistance_map.get(algorithm, QuantumResistance.VULNERABLE)
        
        # Estimate quantum security level
        if quantum_resistance == QuantumResistance.VULNERABLE:
            quantum_security = 0
        elif quantum_resistance == QuantumResistance.PARTIALLY_RESISTANT:
            quantum_security = key_size // 2  # Grover's algorithm impact
        else:
            quantum_security = key_size  # Quantum-safe
        
        asset = CryptographicAsset(
            asset_id=asset_id,
            name=name,
            algorithm=algorithm,
            key_size=key_size,
            usage_context=usage_context,
            current_security_level=key_size if algorithm != CryptographicAlgorithm.ECC else key_size * 2,
            quantum_security_level=quantum_security,
            quantum_resistance=quantum_resistance,
            created_date=datetime.now(),
            migration_priority=migration_priority,
            usage_frequency=np.random.uniform(0.1, 1.0),  # Simulated
            migration_cost_estimate=np.random.uniform(1000, 50000)  # Simulated
        )
        
        # Register with threat analyzer
        self.threat_analyzer.register_cryptographic_asset(asset)
        
        # Save to database
        self._save_crypto_asset(asset)
        
        logger.info(f"Registered cryptographic asset: {name}")
        return asset_id
    
    def _save_crypto_asset(self, asset: CryptographicAsset):
        """Save cryptographic asset to database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO crypto_assets
                (asset_id, name, algorithm, key_size, usage_context,
                 current_security_level, quantum_security_level, quantum_resistance,
                 created_date, expiry_date, last_assessment, migration_priority,
                 migration_planned, replacement_algorithm, usage_frequency,
                 performance_impact, migration_cost_estimate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                asset.asset_id, asset.name, asset.algorithm.value, asset.key_size,
                asset.usage_context, asset.current_security_level, asset.quantum_security_level,
                asset.quantum_resistance.value, asset.created_date.isoformat(),
                asset.expiry_date.isoformat() if asset.expiry_date else None,
                asset.last_assessment.isoformat() if asset.last_assessment else None,
                asset.migration_priority, asset.migration_planned,
                asset.replacement_algorithm.value if asset.replacement_algorithm else None,
                asset.usage_frequency, asset.performance_impact, asset.migration_cost_estimate
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving crypto asset: {e}")
    
    def perform_quantum_readiness_assessment(self) -> Dict[str, Any]:
        """Perform comprehensive quantum readiness assessment"""
        
        # Perform system-wide threat assessment
        threat_assessment = self.threat_analyzer.perform_system_wide_assessment()
        
        # Save assessment to database
        self._save_threat_assessment(threat_assessment)
        
        # Analyze current state
        total_assets = len(self.threat_analyzer.cryptographic_assets)
        vulnerable_assets = threat_assessment.vulnerable_assets_count
        readiness_score = max(0, 100 - (vulnerable_assets / max(1, total_assets) * 100))
        
        # Generate recommendations
        recommendations = self._generate_readiness_recommendations(threat_assessment)
        
        return {
            "assessment_id": threat_assessment.assessment_id,
            "timestamp": threat_assessment.timestamp.isoformat(),
            "readiness_score": readiness_score,
            "threat_level": threat_assessment.threat_level.value,
            "vulnerable_assets": vulnerable_assets,
            "total_assets": total_assets,
            "critical_vulnerabilities": threat_assessment.critical_vulnerabilities,
            "estimated_migration_time_months": threat_assessment.estimated_migration_time,
            "estimated_migration_cost": threat_assessment.estimated_migration_cost,
            "quantum_computer_eta_years": threat_assessment.cryptographically_relevant_quantum_computer_eta,
            "immediate_actions": threat_assessment.immediate_actions,
            "recommendations": recommendations,
            "migration_roadmap": threat_assessment.migration_roadmap,
            "risk_mitigation_strategies": threat_assessment.risk_mitigation_strategies
        }
    
    def _save_threat_assessment(self, assessment: QuantumThreatAssessment):
        """Save threat assessment to database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO quantum_assessments
                (assessment_id, timestamp, threat_level, confidence_level,
                 vulnerable_assets_count, critical_vulnerabilities,
                 estimated_migration_time, estimated_migration_cost,
                 immediate_actions, migration_roadmap, risk_mitigation_strategies)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                assessment.assessment_id, assessment.timestamp.isoformat(),
                assessment.threat_level.value, assessment.confidence_level,
                assessment.vulnerable_assets_count, json.dumps(assessment.critical_vulnerabilities),
                assessment.estimated_migration_time, assessment.estimated_migration_cost,
                json.dumps(assessment.immediate_actions), json.dumps(assessment.migration_roadmap),
                json.dumps(assessment.risk_mitigation_strategies)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving threat assessment: {e}")
    
    def _generate_readiness_recommendations(self, assessment: QuantumThreatAssessment) -> List[str]:
        """Generate quantum readiness recommendations"""
        
        recommendations = []
        
        # Based on threat level
        if assessment.threat_level == QuantumThreatLevel.CRITICAL:
            recommendations.extend([
                "Implement emergency quantum threat response procedures",
                "Prioritize migration of most critical systems immediately",
                "Consider temporary security measures while migrating"
            ])
        elif assessment.threat_level == QuantumThreatLevel.HIGH:
            recommendations.extend([
                "Accelerate post-quantum cryptography adoption",
                "Begin pilot deployments of quantum-safe algorithms",
                "Establish dedicated quantum security team"
            ])
        else:
            recommendations.extend([
                "Develop long-term quantum migration strategy",
                "Stay informed about PQC standardization progress",
                "Plan for crypto-agility in future systems"
            ])
        
        # Based on vulnerable assets count
        if assessment.vulnerable_assets_count > 10:
            recommendations.append("Consider automated migration tools for large-scale deployment")
        
        # Based on cost estimates
        if assessment.estimated_migration_cost > 100000:
            recommendations.append("Seek budget approval for quantum migration project")
            recommendations.append("Consider phased migration to spread costs over time")
        
        return recommendations
    
    def create_migration_plan(self, target_completion_date: datetime,
                            priority_assets: List[str] = None) -> str:
        """Create post-quantum migration plan"""
        
        plan_id = str(uuid.uuid4())
        
        # Get all vulnerable assets
        vulnerable_assets = []
        for asset_id, asset in self.threat_analyzer.cryptographic_assets.items():
            if asset.quantum_resistance in [QuantumResistance.VULNERABLE, QuantumResistance.PARTIALLY_RESISTANT]:
                vulnerable_assets.append(asset_id)
        
        # Prioritize assets
        if priority_assets:
            # Use provided priority list
            prioritized_assets = [aid for aid in priority_assets if aid in vulnerable_assets]
            prioritized_assets.extend([aid for aid in vulnerable_assets if aid not in priority_assets])
        else:
            # Auto-prioritize by risk
            asset_risks = []
            for asset_id in vulnerable_assets:
                vulnerability = self.threat_analyzer.assess_quantum_vulnerability(asset_id)
                asset_risks.append((asset_id, vulnerability["risk_score"]))
            
            asset_risks.sort(key=lambda x: x[1], reverse=True)
            prioritized_assets = [asset_id for asset_id, _ in asset_risks]
        
        # Create migration phases
        assets_per_phase = max(1, len(prioritized_assets) // 4)
        phases = []
        
        for i in range(0, len(prioritized_assets), assets_per_phase):
            phase_assets = prioritized_assets[i:i + assets_per_phase]
            
            phase = {
                "phase_number": len(phases) + 1,
                "name": f"Phase {len(phases) + 1}: Migration Wave",
                "assets": phase_assets,
                "duration_months": len(phase_assets) * 1.5,
                "start_date": (datetime.now() + timedelta(days=len(phases) * 30)).isoformat(),
                "deliverables": [
                    f"Migrate {len(phase_assets)} cryptographic assets",
                    "Conduct security testing",
                    "Update documentation",
                    "Train relevant personnel"
                ],
                "success_criteria": [
                    "All assets migrated successfully",
                    "No security vulnerabilities introduced",
                    "Performance impact within acceptable limits"
                ]
            }
            phases.append(phase)
        
        # Calculate totals
        total_duration = sum(phase["duration_months"] for phase in phases)
        total_cost = sum(
            self.threat_analyzer.cryptographic_assets[asset_id].migration_cost_estimate
            for asset_id in prioritized_assets
        )
        
        # Create migration plan
        migration_plan = PostQuantumMigrationPlan(
            plan_id=plan_id,
            created_date=datetime.now(),
            target_completion_date=target_completion_date,
            phases=phases,
            assets_to_migrate=prioritized_assets,
            migration_priorities={asset_id: "high" if i < len(prioritized_assets) // 3 else "medium" if i < 2 * len(prioritized_assets) // 3 else "low" 
                                for i, asset_id in enumerate(prioritized_assets)},
            estimated_duration_months=int(total_duration),
            estimated_cost=total_cost,
            required_resources=[
                "Cryptography specialists",
                "Security testing team", 
                "System administrators",
                "Training materials",
                "Testing environments"
            ],
            migration_risks=[
                "Performance degradation during migration",
                "Interoperability issues with legacy systems",
                "Staff learning curve for new algorithms",
                "Potential security vulnerabilities during transition"
            ],
            risk_mitigation_measures=[
                "Implement hybrid systems during transition",
                "Comprehensive testing before production deployment",
                "Staff training and certification programs",
                "Rollback procedures for each migration phase"
            ]
        )
        
        # Store migration plan
        self.migration_plans[plan_id] = migration_plan
        self._save_migration_plan(migration_plan)
        
        logger.info(f"Created migration plan: {plan_id}")
        return plan_id
    
    def _save_migration_plan(self, plan: PostQuantumMigrationPlan):
        """Save migration plan to database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO migration_plans
                (plan_id, created_date, target_completion_date, phases,
                 current_phase, assets_to_migrate, migration_priorities,
                 estimated_duration_months, estimated_cost, required_resources,
                 completion_percentage, milestones_completed, migration_risks,
                 risk_mitigation_measures)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                plan.plan_id, plan.created_date.isoformat(),
                plan.target_completion_date.isoformat(), json.dumps(plan.phases),
                plan.current_phase, json.dumps(plan.assets_to_migrate),
                json.dumps(plan.migration_priorities), plan.estimated_duration_months,
                plan.estimated_cost, json.dumps(plan.required_resources),
                plan.completion_percentage, json.dumps(plan.milestones_completed),
                json.dumps(plan.migration_risks), json.dumps(plan.risk_mitigation_measures)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving migration plan: {e}")
    
    def get_quantum_readiness_dashboard_data(self) -> Dict[str, Any]:
        """Get data for quantum readiness dashboard"""
        
        dashboard_data = {
            "overview": {},
            "threat_assessment": {},
            "asset_analysis": {},
            "migration_status": {},
            "timeline": {}
        }
        
        try:
            # Overview metrics
            total_assets = len(self.threat_analyzer.cryptographic_assets)
            vulnerable_assets = len([
                asset for asset in self.threat_analyzer.cryptographic_assets.values()
                if asset.quantum_resistance in [QuantumResistance.VULNERABLE, QuantumResistance.PARTIALLY_RESISTANT]
            ])
            
            dashboard_data["overview"] = {
                "total_cryptographic_assets": total_assets,
                "vulnerable_assets": vulnerable_assets,
                "quantum_safe_assets": total_assets - vulnerable_assets,
                "readiness_percentage": max(0, 100 - (vulnerable_assets / max(1, total_assets) * 100)),
                "migration_plans": len(self.migration_plans)
            }
            
            # Asset analysis
            algorithm_distribution = {}
            resistance_distribution = {}
            
            for asset in self.threat_analyzer.cryptographic_assets.values():
                algo = asset.algorithm.value
                resistance = asset.quantum_resistance.value
                
                algorithm_distribution[algo] = algorithm_distribution.get(algo, 0) + 1
                resistance_distribution[resistance] = resistance_distribution.get(resistance, 0) + 1
            
            dashboard_data["asset_analysis"] = {
                "algorithm_distribution": algorithm_distribution,
                "resistance_distribution": resistance_distribution
            }
            
            # Get latest threat assessment
            if self.threat_analyzer.threat_assessments:
                latest_assessment = self.threat_analyzer.threat_assessments[-1]
                dashboard_data["threat_assessment"] = {
                    "threat_level": latest_assessment.threat_level.value,
                    "confidence_level": latest_assessment.confidence_level,
                    "quantum_computer_eta": latest_assessment.cryptographically_relevant_quantum_computer_eta,
                    "critical_vulnerabilities": len(latest_assessment.critical_vulnerabilities),
                    "estimated_migration_cost": latest_assessment.estimated_migration_cost
                }
            
            # Migration status
            if self.migration_plans:
                active_plans = len([p for p in self.migration_plans.values() if p.completion_percentage < 100])
                avg_completion = np.mean([p.completion_percentage for p in self.migration_plans.values()]) if self.migration_plans else 0
                
                dashboard_data["migration_status"] = {
                    "active_plans": active_plans,
                    "average_completion": avg_completion,
                    "total_estimated_cost": sum(p.estimated_cost for p in self.migration_plans.values())
                }
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {e}")
        
        return dashboard_data

# Streamlit Integration Functions

def initialize_quantum_readiness():
    """Initialize quantum readiness system"""
    if 'quantum_readiness_manager' not in st.session_state:
        st.session_state.quantum_readiness_manager = QuantumReadinessManager()
        
        # Create sample cryptographic assets
        manager = st.session_state.quantum_readiness_manager
        
        sample_assets = [
            ("RSA Certificate Authority", CryptographicAlgorithm.RSA, 2048, "SSL/TLS certificates", "critical"),
            ("ECDSA API Signing", CryptographicAlgorithm.ECC, 256, "API authentication", "high"),
            ("AES Data Encryption", CryptographicAlgorithm.AES, 256, "Database encryption", "medium"),
            ("SHA-256 Hash Function", CryptographicAlgorithm.SHA256, 256, "Data integrity", "medium"),
            ("RSA VPN Gateway", CryptographicAlgorithm.RSA, 4096, "VPN authentication", "high")
        ]
        
        for name, algorithm, key_size, context, priority in sample_assets:
            manager.register_cryptographic_asset(name, algorithm, key_size, context, priority)
    
    return st.session_state.quantum_readiness_manager

def render_quantum_readiness_dashboard():
    """Render quantum computing preparation dashboard"""
    st.header("🔮 Quantum Computing Preparation & Post-Quantum Cryptography")
    
    quantum_manager = initialize_quantum_readiness()
    
    # Get dashboard data
    dashboard_data = quantum_manager.get_quantum_readiness_dashboard_data()
    
    # Overview metrics
    overview = dashboard_data.get("overview", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_assets = overview.get("total_cryptographic_assets", 0)
        st.metric("Crypto Assets", total_assets)
    
    with col2:
        vulnerable = overview.get("vulnerable_assets", 0)
        st.metric("Vulnerable Assets", vulnerable)
    
    with col3:
        readiness = overview.get("readiness_percentage", 0)
        st.metric("Quantum Readiness", f"{readiness:.1f}%")
    
    with col4:
        migration_plans = overview.get("migration_plans", 0)
        st.metric("Migration Plans", migration_plans)
    
    # Readiness status indicator
    if readiness >= 80:
        st.success("🟢 Excellent quantum readiness - well prepared for quantum threats")
    elif readiness >= 60:
        st.warning("🟡 Good quantum readiness - some improvements needed")
    elif readiness >= 40:
        st.warning("🟠 Moderate quantum readiness - significant work required")
    else:
        st.error("🔴 Poor quantum readiness - urgent action needed")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Assessment",
        "🔐 Crypto Assets",
        "⚡ Quantum Attacks",
        "🛡️ Post-Quantum Crypto",
        "🗺️ Migration Planning",
        "📈 Monitoring"
    ])
    
    with tab1:
        st.subheader("Quantum Threat Assessment")
        
        # Run assessment button
        if st.button("🔍 Run Quantum Readiness Assessment"):
            with st.spinner("Analyzing quantum threats and readiness..."):
                assessment_results = quantum_manager.perform_quantum_readiness_assessment()
                
                st.success("Assessment completed!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Assessment Summary:**")
                    st.write(f"• **Readiness Score:** {assessment_results['readiness_score']:.1f}%")
                    st.write(f"• **Threat Level:** {assessment_results['threat_level'].title()}")
                    st.write(f"• **Vulnerable Assets:** {assessment_results['vulnerable_assets']}/{assessment_results['total_assets']}")
                    st.write(f"• **Migration Time:** {assessment_results['estimated_migration_time_months']} months")
                    st.write(f"• **Migration Cost:** ${assessment_results['estimated_migration_cost']:,.2f}")
                    st.write(f"• **Quantum Computer ETA:** {assessment_results['quantum_computer_eta_years']} years")
                
                with col2:
                    if assessment_results.get("critical_vulnerabilities"):
                        st.write("**Critical Vulnerabilities:**")
                        for vuln in assessment_results["critical_vulnerabilities"][:5]:
                            st.write(f"⚠️ {vuln}")
                    else:
                        st.write("✅ No critical vulnerabilities found")
                
                # Immediate actions
                if assessment_results.get("immediate_actions"):
                    st.write("**Immediate Actions Required:**")
                    for action in assessment_results["immediate_actions"]:
                        st.write(f"🚨 {action}")
                
                # Recommendations
                if assessment_results.get("recommendations"):
                    st.write("**Recommendations:**")
                    for rec in assessment_results["recommendations"]:
                        st.write(f"💡 {rec}")
        
        # Threat assessment visualization
        threat_data = dashboard_data.get("threat_assessment", {})
        
        if threat_data:
            st.subheader("Threat Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Threat level gauge
                threat_level = threat_data.get("threat_level", "minimal")
                threat_levels = ["minimal", "low", "moderate", "high", "critical"]
                threat_score = (threat_levels.index(threat_level) + 1) * 20
                
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = threat_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Quantum Threat Level"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 20], 'color': "lightgray"},
                            {'range': [20, 40], 'color': "yellow"},
                            {'range': [40, 60], 'color': "orange"},
                            {'range': [60, 80], 'color': "red"},
                            {'range': [80, 100], 'color': "darkred"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                st.write("**Threat Details:**")
                st.write(f"• **Current Level:** {threat_level.title()}")
                st.write(f"• **Confidence:** {threat_data.get('confidence_level', 0)*100:.1f}%")
                st.write(f"• **Quantum Computer ETA:** {threat_data.get('quantum_computer_eta', 'Unknown')} years")
                st.write(f"• **Critical Vulnerabilities:** {threat_data.get('critical_vulnerabilities', 0)}")
                st.write(f"• **Est. Migration Cost:** ${threat_data.get('estimated_migration_cost', 0):,.2f}")
    
    with tab2:
        st.subheader("Cryptographic Assets Inventory")
        
        # Asset management actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("➕ Add New Asset"):
                st.session_state.show_add_asset_form = True
        
        with col2:
            if st.button("🔄 Refresh Assets"):
                st.rerun()
        
        with col3:
            if st.button("📊 Analyze All Assets"):
                with st.spinner("Analyzing all cryptographic assets..."):
                    analysis_results = []
                    
                    for asset_id in quantum_manager.threat_analyzer.cryptographic_assets.keys():
                        result = quantum_manager.threat_analyzer.assess_quantum_vulnerability(asset_id)
                        analysis_results.append(result)
                    
                    st.success(f"Analyzed {len(analysis_results)} assets")
                    st.session_state.asset_analysis_results = analysis_results
        
        # Add asset form
        if hasattr(st.session_state, 'show_add_asset_form') and st.session_state.show_add_asset_form:
            with st.expander("Add New Cryptographic Asset", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    asset_name = st.text_input("Asset Name")
                    algorithm = st.selectbox("Algorithm", [alg.value for alg in CryptographicAlgorithm])
                    key_size = st.number_input("Key Size (bits)", min_value=128, max_value=8192, value=2048)
                
                with col2:
                    usage_context = st.text_area("Usage Context")
                    migration_priority = st.selectbox("Migration Priority", ["low", "medium", "high", "critical"])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("✅ Add Asset"):
                        if asset_name and algorithm:
                            asset_id = quantum_manager.register_cryptographic_asset(
                                asset_name, CryptographicAlgorithm(algorithm), 
                                key_size, usage_context, migration_priority
                            )
                            st.success(f"Added asset: {asset_name}")
                            st.session_state.show_add_asset_form = False
                            st.rerun()
                        else:
                            st.error("Please provide asset name and algorithm")
                
                with col2:
                    if st.button("❌ Cancel"):
                        st.session_state.show_add_asset_form = False
                        st.rerun()
        
        # Assets list
        assets = quantum_manager.threat_analyzer.cryptographic_assets
        
        if assets:
            st.write(f"**Registered Assets ({len(assets)}):**")
            
            # Create assets DataFrame
            asset_data = []
            for asset in assets.values():
                # Get vulnerability assessment
                vulnerability = quantum_manager.threat_analyzer.assess_quantum_vulnerability(asset.asset_id)
                
                resistance_icons = {
                    "vulnerable": "🔴",
                    "partially_resistant": "🟡", 
                    "quantum_safe": "🟢",
                    "quantum_proven": "🟢"
                }
                
                asset_data.append({
                    'Asset Name': asset.name,
                    'Algorithm': asset.algorithm.value.upper(),
                    'Key Size': f"{asset.key_size} bits",
                    'Quantum Resistance': f"{resistance_icons.get(asset.quantum_resistance.value, '⚪')} {asset.quantum_resistance.value.replace('_', ' ').title()}",
                    'Risk Level': vulnerability.get('risk_level', 'unknown').title(),
                    'Migration Priority': asset.migration_priority.title(),
                    'Usage Context': asset.usage_context[:30] + '...' if len(asset.usage_context) > 30 else asset.usage_context
                })
            
            assets_df = pd.DataFrame(asset_data)
            st.dataframe(assets_df, use_container_width=True)
            
            # Asset analysis results
            if hasattr(st.session_state, 'asset_analysis_results'):
                st.subheader("Vulnerability Analysis Results")
                
                for result in st.session_state.asset_analysis_results[:5]:  # Show top 5
                    with st.expander(f"🔍 {result['asset_name']} - {result['risk_level'].title()} Risk"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Algorithm:** {result['algorithm'].upper()}")
                            st.write(f"**Quantum Resistance:** {result['quantum_resistance'].replace('_', ' ').title()}")
                            st.write(f"**Break Probability:** {result['break_probability']*100:.1f}%")
                            st.write(f"**Risk Score:** {result['risk_score']:.1f}/100")
                        
                        with col2:
                            st.write(f"**Migration Urgency:** {result['migration_urgency'].title()}")
                            
                            if result.get('recommended_actions'):
                                st.write("**Recommended Actions:**")
                                for action in result['recommended_actions'][:3]:
                                    st.write(f"• {action}")
        
        # Asset distribution charts
        if assets:
            col1, col2 = st.columns(2)
            
            with col1:
                # Algorithm distribution
                algo_counts = {}
                for asset in assets.values():
                    algo = asset.algorithm.value
                    algo_counts[algo] = algo_counts.get(algo, 0) + 1
                
                if algo_counts:
                    algo_df = pd.DataFrame(list(algo_counts.items()), columns=['Algorithm', 'Count'])
                    fig_algo = px.pie(algo_df, values='Count', names='Algorithm',
                                    title='Cryptographic Algorithms Distribution')
                    st.plotly_chart(fig_algo, use_container_width=True)
            
            with col2:
                # Quantum resistance distribution
                resistance_counts = {}
                for asset in assets.values():
                    resistance = asset.quantum_resistance.value
                    resistance_counts[resistance] = resistance_counts.get(resistance, 0) + 1
                
                if resistance_counts:
                    resist_df = pd.DataFrame(list(resistance_counts.items()), columns=['Resistance Level', 'Count'])
                    fig_resist = px.bar(resist_df, x='Resistance Level', y='Count',
                                      title='Quantum Resistance Distribution')
                    st.plotly_chart(fig_resist, use_container_width=True)
        else:
            st.info("No cryptographic assets registered. Click 'Add New Asset' to get started.")
    
    with tab3:
        st.subheader("Quantum Attack Simulation")
        
        st.write("Simulate quantum attacks against different cryptographic algorithms to understand vulnerabilities.")
        
        # Attack simulation interface
        col1, col2, col3 = st.columns(3)
        
        with col1:
            attack_algorithm = st.selectbox(
                "Target Algorithm",
                [alg.value for alg in CryptographicAlgorithm if alg in [
                    CryptographicAlgorithm.RSA, CryptographicAlgorithm.ECC, 
                    CryptographicAlgorithm.AES, CryptographicAlgorithm.SHA256
                ]]
            )
        
        with col2:
            key_size = st.selectbox(
                "Key Size (bits)",
                [1024, 2048, 3072, 4096] if attack_algorithm == "rsa" else
                [128, 192, 256, 384, 521] if attack_algorithm == "ecc" else
                [128, 192, 256]
            )
        
        with col3:
            if st.button("⚡ Simulate Quantum Attack"):
                with st.spinner("Simulating quantum attack..."):
                    attack_result = quantum_manager.pqc_engine.simulate_quantum_attack(
                        CryptographicAlgorithm(attack_algorithm), key_size
                    )
                    
                    st.write("**Attack Simulation Results:**")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Target:** {attack_result['target_algorithm'].upper()} {key_size}-bit")
                        st.write(f"**Attack Method:** {attack_result['attack_method']}")
                        st.write(f"**Quantum Speedup:** {attack_result['quantum_speedup'].title()}")
                        st.write(f"**Qubits Required:** {attack_result.get('qubits_required', 'N/A'):,}")
                    
                    with col2:
                        st.write(f"**Original Security:** {attack_result.get('original_security_bits', key_size)} bits")
                        st.write(f"**Effective Security:** {attack_result.get('effective_security_bits', 0)} bits")
                        st.write(f"**Vulnerability Level:** {attack_result.get('vulnerability_level', 'unknown').title()}")
                        
                        break_time = attack_result.get('estimated_break_time_years', float('inf'))
                        if break_time == float('inf'):
                            st.write("**Break Time:** Computationally infeasible")
                        elif break_time < 0.001:
                            st.write("**Break Time:** < 1 day")
                        elif break_time < 1:
                            st.write(f"**Break Time:** {break_time*365:.0f} days")
                        else:
                            st.write(f"**Break Time:** {break_time:.1f} years")
                    
                    # Vulnerability assessment
                    vulnerability_level = attack_result.get('vulnerability_level', 'unknown')
                    
                    if vulnerability_level == 'critical':
                        st.error("🔴 CRITICAL: This algorithm is completely vulnerable to quantum attacks!")
                    elif vulnerability_level == 'high':
                        st.warning("🟠 HIGH: This algorithm has significant quantum vulnerabilities!")
                    elif vulnerability_level == 'moderate':
                        st.warning("🟡 MODERATE: This algorithm has some quantum resistance but may need upgrading")
                    else:
                        st.success("🟢 LOW: This algorithm appears to be quantum-resistant")
        
        # Quantum computing timeline
        st.subheader("Quantum Computing Timeline")
        
        timeline_data = {
            'Year': [2024, 2026, 2028, 2030, 2032, 2035, 2040],
            'Scenario': ['Conservative', 'Conservative', 'Moderate', 'Moderate', 'Aggressive', 'Aggressive', 'Breakthrough'],
            'Expected Capabilities': [
                'Current NISQ systems',
                'Improved error correction',
                'Fault-tolerant qubits (100s)',
                'Small-scale algorithms',
                'Break RSA-1024',
                'Break RSA-2048',
                'Break all classical crypto'
            ],
            'Threat Level': [1, 2, 3, 4, 6, 8, 10]
        }
        
        timeline_df = pd.DataFrame(timeline_data)
        
        fig_timeline = px.line(timeline_df, x='Year', y='Threat Level', 
                              hover_data=['Scenario', 'Expected Capabilities'],
                              title='Quantum Computing Threat Timeline',
                              markers=True)
        
        fig_timeline.add_hline(y=5, line_dash="dash", line_color="red", 
                              annotation_text="Critical Threat Level")
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Quantum supremacy milestones
        st.subheader("Quantum Computing Milestones")
        
        milestones = [
            {"Year": "2019", "Event": "Google's Quantum Supremacy", "Significance": "First demonstration of quantum advantage"},
            {"Year": "2021", "Event": "IBM 127-qubit Eagle", "Significance": "Breakthrough in qubit scaling"},
            {"Year": "2023", "Event": "Error correction progress", "Significance": "Improved quantum error rates"},
            {"Year": "2024", "Event": "Commercial quantum cloud", "Significance": "Accessible quantum computing"},
            {"Year": "~2030", "Event": "Cryptographically relevant QC", "Significance": "Threat to current cryptography"}
        ]
        
        for milestone in milestones:
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 3])
                
                with col1:
                    st.write(f"**{milestone['Year']}**")
                
                with col2:
                    st.write(milestone['Event'])
                
                with col3:
                    st.write(milestone['Significance'])
                
                st.markdown("---")
    
    with tab4:
        st.subheader("Post-Quantum Cryptography")
        
        st.write("Explore quantum-safe cryptographic algorithms and their properties.")
        
        # PQC algorithm information
        pqc_algorithms = {
            "CRYSTALS-Kyber": {
                "type": "Key Encapsulation Mechanism",
                "security_basis": "Lattice-based (Module-LWE)",
                "nist_status": "Selected Standard",
                "key_sizes": "1568 bytes (public key)",
                "performance": "Fast encryption/decryption",
                "use_cases": "TLS, VPN, secure messaging"
            },
            "CRYSTALS-Dilithium": {
                "type": "Digital Signatures", 
                "security_basis": "Lattice-based (Module-LWE/SIS)",
                "nist_status": "Selected Standard",
                "key_sizes": "1312 bytes (public key)",
                "performance": "Medium signing speed",
                "use_cases": "Code signing, certificates, authentication"
            },
            "SPHINCS+": {
                "type": "Digital Signatures",
                "security_basis": "Hash-based",
                "nist_status": "Selected Standard", 
                "key_sizes": "32 bytes (public key)",
                "performance": "Slow signing, fast verification",
                "use_cases": "Long-term signatures, certificates"
            },
            "FALCON": {
                "type": "Digital Signatures",
                "security_basis": "Lattice-based (NTRU)",
                "nist_status": "Alternative Standard",
                "key_sizes": "897 bytes (public key)",
                "performance": "Fast signing and verification",
                "use_cases": "Constrained environments"
            }
        }
        
        # Algorithm selector
        selected_algorithm = st.selectbox(
            "Select Post-Quantum Algorithm",
            list(pqc_algorithms.keys())
        )
        
        if selected_algorithm:
            algo_info = pqc_algorithms[selected_algorithm]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Algorithm Details:**")
                for key, value in algo_info.items():
                    st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
            
            with col2:
                # Generate sample key
                if st.button(f"🔑 Generate {selected_algorithm} Key"):
                    with st.spinner("Generating quantum-safe cryptographic key..."):
                        # Map to internal algorithm enum
                        algo_map = {
                            "CRYSTALS-Kyber": CryptographicAlgorithm.KYBER,
                            "CRYSTALS-Dilithium": CryptographicAlgorithm.DILITHIUM,
                            "SPHINCS+": CryptographicAlgorithm.SPHINCS
                        }
                        
                        if selected_algorithm in algo_map:
                            key_data = quantum_manager.pqc_engine.generate_quantum_safe_key(
                                algo_map[selected_algorithm],
                                f"demo_{selected_algorithm.lower()}_{int(time.time())}"
                            )
                            
                            st.success("Quantum-safe key generated!")
                            
                            with st.expander("View Key Details"):
                                st.json({
                                    "key_id": key_data["key_id"],
                                    "algorithm": key_data["algorithm"],
                                    "key_size": key_data["key_size"],
                                    "security_level": key_data["security_level"],
                                    "performance_factor": key_data["performance_factor"],
                                    "created_date": key_data["created_date"]
                                })
        
        # Performance comparison
        st.subheader("Performance Comparison")
        
        if st.button("📊 Benchmark PQC Algorithms"):
            with st.spinner("Benchmarking post-quantum algorithms..."):
                benchmark_results = []
                
                pqc_algos = [
                    CryptographicAlgorithm.KYBER,
                    CryptographicAlgorithm.DILITHIUM,
                    CryptographicAlgorithm.SPHINCS
                ]
                
                for algo in pqc_algos:
                    result = quantum_manager.pqc_engine.benchmark_pqc_performance(algo)
                    if "error" not in result:
                        benchmark_results.append(result)
                
                if benchmark_results:
                    # Create comparison DataFrame
                    comparison_data = []
                    for result in benchmark_results:
                        comparison_data.append({
                            'Algorithm': result['algorithm'].upper(),
                            'Key Gen (ms)': f"{result['key_generation_time_ms']:.2f}",
                            'Encrypt (ms)': f"{result['encryption_time_ms']:.2f}",
                            'Decrypt (ms)': f"{result['decryption_time_ms']:.2f}",
                            'Key Size (KB)': f"{result['key_size_bytes'] / 1024:.1f}",
                            'Security (bits)': result['security_level_bits'],
                            'Overhead (%)': f"{result['performance_overhead']:.1f}%"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Performance charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Key generation time
                        key_gen_data = [(r['algorithm'], r['key_generation_time_ms']) for r in benchmark_results]
                        key_gen_df = pd.DataFrame(key_gen_data, columns=['Algorithm', 'Time (ms)'])
                        
                        fig_keygen = px.bar(key_gen_df, x='Algorithm', y='Time (ms)',
                                          title='Key Generation Time Comparison')
                        st.plotly_chart(fig_keygen, use_container_width=True)
                    
                    with col2:
                        # Key size comparison
                        key_size_data = [(r['algorithm'], r['key_size_bytes'] / 1024) for r in benchmark_results]
                        key_size_df = pd.DataFrame(key_size_data, columns=['Algorithm', 'Size (KB)'])
                        
                        fig_keysize = px.bar(key_size_df, x='Algorithm', y='Size (KB)',
                                           title='Public Key Size Comparison')
                        st.plotly_chart(fig_keysize, use_container_width=True)
        
        # Hybrid cryptography
        st.subheader("Hybrid Cryptographic Systems")
        
        st.write("Combine classical and post-quantum algorithms for transition security.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            classical_algo = st.selectbox(
                "Classical Algorithm",
                ["rsa", "ecc", "aes"]
            )
        
        with col2:
            pq_algo = st.selectbox(
                "Post-Quantum Algorithm", 
                ["kyber", "dilithium", "sphincs"]
            )
        
        with col3:
            if st.button("🔗 Create Hybrid System"):
                hybrid_system = quantum_manager.pqc_engine.create_hybrid_system(
                    CryptographicAlgorithm(classical_algo),
                    CryptographicAlgorithm(pq_algo),
                    f"hybrid_{classical_algo}_{pq_algo}_{int(time.time())}"
                )
                
                st.success("Hybrid cryptographic system created!")
                
                with st.expander("View Hybrid System Details"):
                    st.json(hybrid_system)
    
    with tab5:
        st.subheader("Migration Planning")
        
        # Migration plan creation
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Create Migration Plan:**")
            
            target_date = st.date_input(
                "Target Completion Date",
                value=datetime.now().date() + timedelta(days=365)
            )
            
            # Priority assets selection
            available_assets = list(quantum_manager.threat_analyzer.cryptographic_assets.keys())
            priority_assets = st.multiselect(
                "Priority Assets (optional)",
                available_assets,
                help="Select assets to prioritize in migration. If none selected, assets will be auto-prioritized by risk."
            )
            
            if st.button("📋 Create Migration Plan"):
                if available_assets:
                    with st.spinner("Creating post-quantum migration plan..."):
                        plan_id = quantum_manager.create_migration_plan(
                            datetime.combine(target_date, datetime.min.time()),
                            priority_assets if priority_assets else None
                        )
                        
                        st.success(f"Migration plan created: {plan_id}")
                        st.session_state.current_migration_plan = plan_id
                        st.rerun()
                else:
                    st.warning("No cryptographic assets available for migration planning.")
        
        with col2:
            # Display existing migration plans
            if quantum_manager.migration_plans:
                st.write("**Existing Migration Plans:**")
                
                for plan_id, plan in quantum_manager.migration_plans.items():
                    with st.container():
                        completion = plan.completion_percentage
                        progress_color = "🟢" if completion >= 80 else "🟡" if completion >= 50 else "🔴"
                        
                        st.write(f"{progress_color} **Plan:** {plan_id[:8]}...")
                        st.write(f"• **Progress:** {completion:.1f}%")
                        st.write(f"• **Target Date:** {plan.target_completion_date.strftime('%Y-%m-%d')}")
                        st.write(f"• **Assets:** {len(plan.assets_to_migrate)}")
                        st.write(f"• **Cost:** ${plan.estimated_cost:,.2f}")
                        
                        if st.button(f"📖 View Details", key=f"view_{plan_id}"):
                            st.session_state.selected_migration_plan = plan_id
                        
                        st.markdown("---")
            else:
                st.info("No migration plans created yet.")
        
        # Display selected migration plan details
        if hasattr(st.session_state, 'selected_migration_plan'):
            plan_id = st.session_state.selected_migration_plan
            
            if plan_id in quantum_manager.migration_plans:
                plan = quantum_manager.migration_plans[plan_id]
                
                st.subheader(f"Migration Plan Details: {plan_id[:8]}...")
                
                # Plan overview
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Assets to Migrate", len(plan.assets_to_migrate))
                
                with col2:
                    st.metric("Estimated Duration", f"{plan.estimated_duration_months} months")
                
                with col3:
                    st.metric("Estimated Cost", f"${plan.estimated_cost:,.2f}")
                
                with col4:
                    st.metric("Completion", f"{plan.completion_percentage:.1f}%")
                
                # Migration phases
                st.write("**Migration Phases:**")
                
                for i, phase in enumerate(plan.phases):
                    phase_status = "✅" if i < plan.current_phase else "🔄" if i == plan.current_phase else "⏳"
                    
                    with st.expander(f"{phase_status} {phase.get('name', f'Phase {i+1}')}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Phase Number:** {phase.get('phase_number', i+1)}")
                            st.write(f"**Duration:** {phase.get('duration_months', 0):.1f} months")
                            st.write(f"**Assets:** {len(phase.get('assets', []))}")
                            if phase.get('start_date'):
                                st.write(f"**Start Date:** {phase['start_date'][:10]}")
                        
                        with col2:
                            if phase.get('deliverables'):
                                st.write("**Deliverables:**")
                                for deliverable in phase['deliverables']:
                                    st.write(f"• {deliverable}")
                            
                            if phase.get('success_criteria'):
                                st.write("**Success Criteria:**")
                                for criteria in phase['success_criteria']:
                                    st.write(f"• {criteria}")
                
                # Risks and mitigation
                col1, col2 = st.columns(2)
                
                with col1:
                    if plan.migration_risks:
                        st.write("**Migration Risks:**")
                        for risk in plan.migration_risks:
                            st.write(f"⚠️ {risk}")
                
                with col2:
                    if plan.risk_mitigation_measures:
                        st.write("**Risk Mitigation:**")
                        for measure in plan.risk_mitigation_measures:
                            st.write(f"🛡️ {measure}")
                
                # Progress simulation
                if st.button("📈 Simulate Progress Update"):
                    # Simulate some progress
                    plan.completion_percentage = min(100, plan.completion_percentage + np.random.uniform(5, 15))
                    if plan.completion_percentage > (plan.current_phase + 1) * (100 / len(plan.phases)):
                        plan.current_phase = min(len(plan.phases) - 1, plan.current_phase + 1)
                    
                    st.success(f"Progress updated: {plan.completion_percentage:.1f}%")
                    st.rerun()
        
        # Migration timeline visualization
        if quantum_manager.migration_plans:
            st.subheader("Migration Timeline")
            
            timeline_data = []
            for plan_id, plan in quantum_manager.migration_plans.items():
                for i, phase in enumerate(plan.phases):
                    start_date = datetime.fromisoformat(phase.get('start_date', datetime.now().isoformat()))
                    end_date = start_date + timedelta(days=phase.get('duration_months', 1) * 30)
                    
                    timeline_data.append({
                        'Plan': plan_id[:8],
                        'Phase': phase.get('name', f'Phase {i+1}'),
                        'Start': start_date,
                        'End': end_date,
                        'Duration': phase.get('duration_months', 1)
                    })
            
            if timeline_data:
                timeline_df = pd.DataFrame(timeline_data)
                
                fig_timeline = px.timeline(
                    timeline_df, 
                    x_start='Start', 
                    x_end='End',
                    y='Plan', 
                    color='Phase',
                    title='Migration Timeline'
                )
                
                st.plotly_chart(fig_timeline, use_container_width=True)
    
    with tab6:
        st.subheader("Quantum Readiness Monitoring")
        
        # Real-time monitoring dashboard
        col1, col2 = st.columns(2)
        
        with col1:
            # Readiness score over time (simulated)
            if st.button("📊 Update Readiness Metrics"):
                # Simulate readiness tracking
                dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
                readiness_scores = [50, 52, 55, 58, 62, 65, 68, 72, 75, 78, 82, 85]
                
                readiness_df = pd.DataFrame({
                    'Date': dates,
                    'Readiness Score': readiness_scores
                })
                
                fig_readiness = px.line(readiness_df, x='Date', y='Readiness Score',
                                      title='Quantum Readiness Progress',
                                      markers=True)
                fig_readiness.add_hline(y=80, line_dash="dash", line_color="green",
                                      annotation_text="Target Readiness Level")
                
                st.plotly_chart(fig_readiness, use_container_width=True)
        
        with col2:
            # Threat level monitoring
            threat_history = [
                {'Month': 'Jan 2024', 'Threat Level': 'Low', 'Score': 2},
                {'Month': 'Apr 2024', 'Threat Level': 'Low', 'Score': 2},
                {'Month': 'Jul 2024', 'Threat Level': 'Moderate', 'Score': 3},
                {'Month': 'Oct 2024', 'Threat Level': 'Moderate', 'Score': 3},
                {'Month': 'Jan 2025', 'Threat Level': 'Moderate', 'Score': 4}
            ]
            
            threat_df = pd.DataFrame(threat_history)
            
            fig_threat = px.line(threat_df, x='Month', y='Score',
                               title='Quantum Threat Level Trend',
                               markers=True,
                               hover_data=['Threat Level'])
            
            st.plotly_chart(fig_threat, use_container_width=True)
        
        # Key performance indicators
        st.subheader("Key Performance Indicators")
        
        kpis = [
            {"KPI": "Quantum Readiness Score", "Current": "85%", "Target": "90%", "Status": "🟡"},
            {"KPI": "Vulnerable Assets", "Current": "3", "Target": "0", "Status": "🟠"},
            {"KPI": "Migration Progress", "Current": "60%", "Target": "100%", "Status": "🟢"},
            {"KPI": "Staff Training Completion", "Current": "75%", "Target": "100%", "Status": "🟡"},
            {"KPI": "Budget Utilization", "Current": "45%", "Target": "80%", "Status": "🟢"},
            {"KPI": "Compliance Gaps", "Current": "2", "Target": "0", "Status": "🟠"}
        ]
        
        kpi_df = pd.DataFrame(kpis)
        st.dataframe(kpi_df, use_container_width=True, hide_index=True)
        
        # Monitoring alerts
        st.subheader("Monitoring Alerts")
        
        alerts = [
            {"Timestamp": "2024-12-13 14:30", "Level": "🟡 Warning", "Message": "RSA-2048 certificate expires in 90 days - plan migration"},
            {"Timestamp": "2024-12-13 09:15", "Level": "🔴 Critical", "Message": "New quantum computing breakthrough reported - reassess timeline"},
            {"Timestamp": "2024-12-12 16:45", "Level": "🟢 Info", "Message": "CRYSTALS-Kyber implementation successfully tested"},
            {"Timestamp": "2024-12-12 11:20", "Level": "🟡 Warning", "Message": "Migration Phase 2 behind schedule by 2 weeks"},
            {"Timestamp": "2024-12-11 08:30", "Level": "🟢 Info", "Message": "Staff training completion rate improved to 75%"}
        ]
        
        for alert in alerts:
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 4])
                
                with col1:
                    st.write(alert["Timestamp"])
                
                with col2:
                    st.write(alert["Level"])
                
                with col3:
                    st.write(alert["Message"])
                
                st.markdown("---")
        
        # Export monitoring data
        if st.button("📊 Export Monitoring Report"):
            monitoring_data = {
                "quantum_readiness_assessment": dashboard_data,
                "kpis": kpis,
                "alerts": alerts,
                "export_timestamp": datetime.now().isoformat()
            }
            
            report_json = json.dumps(monitoring_data, indent=2, default=str)
            
            st.download_button(
                label="Download Monitoring Report",
                data=report_json,
                file_name=f"quantum_monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime='application/json'
            )

if __name__ == "__main__":
    # Example usage and testing
    
    print("Testing quantum computing preparation and post-quantum cryptography...")
    
    # Initialize quantum readiness manager
    quantum_manager = QuantumReadinessManager()
    
    # Register sample cryptographic assets
    sample_assets = [
        ("SSL Certificate", CryptographicAlgorithm.RSA, 2048, "Web server security", "high"),
        ("API Signing Key", CryptographicAlgorithm.ECC, 256, "API authentication", "medium"),
        ("Database Encryption", CryptographicAlgorithm.AES, 256, "Data at rest", "critical")
    ]
    
    for name, algorithm, key_size, context, priority in sample_assets:
        asset_id = quantum_manager.register_cryptographic_asset(name, algorithm, key_size, context, priority)
        print(f"Registered asset: {name} ({asset_id})")
    
    # Perform quantum readiness assessment
    assessment = quantum_manager.perform_quantum_readiness_assessment()
    print(f"\nQuantum Readiness Assessment:")
    print(f"- Readiness Score: {assessment['readiness_score']:.1f}%")
    print(f"- Threat Level: {assessment['threat_level']}")
    print(f"- Vulnerable Assets: {assessment['vulnerable_assets']}/{assessment['total_assets']}")
    print(f"- Migration Cost: ${assessment['estimated_migration_cost']:,.2f}")
    
    # Create migration plan
    target_date = datetime.now() + timedelta(days=365)
    plan_id = quantum_manager.create_migration_plan(target_date)
    print(f"\nCreated migration plan: {plan_id}")
    
    # Test post-quantum cryptography
    print(f"\nTesting Post-Quantum Cryptography:")
    
    # Generate quantum-safe key
    key_data = quantum_manager.pqc_engine.generate_quantum_safe_key(
        CryptographicAlgorithm.KYBER, "test_kyber_key"
    )
    print(f"Generated Kyber key: {key_data['key_size']} bytes")
    
    # Simulate quantum attack
    attack_result = quantum_manager.pqc_engine.simulate_quantum_attack(
        CryptographicAlgorithm.RSA, 2048
    )
    print(f"Quantum attack simulation on RSA-2048:")
    print(f"- Attack method: {attack_result['attack_method']}")
    print(f"- Vulnerability: {attack_result['vulnerability_level']}")
    print(f"- Break time: {attack_result.get('estimated_break_time_years', 'N/A')} years")
    
    # Create hybrid system
    hybrid_system = quantum_manager.pqc_engine.create_hybrid_system(
        CryptographicAlgorithm.RSA,
        CryptographicAlgorithm.KYBER,
        "hybrid_rsa_kyber"
    )
    print(f"Created hybrid system: {hybrid_system['system_id']}")
    
    print("Quantum computing preparation and post-quantum cryptography test completed!")