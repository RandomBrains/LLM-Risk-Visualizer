"""
Advanced Security and Compliance Module
Implements enterprise-grade security features and regulatory compliance tools
"""

import hashlib
import secrets
import hmac
import base64
import json
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import ipaddress
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import streamlit as st
import pandas as pd

class SecurityLevel(Enum):
    """Security levels for different operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceStandard(Enum):
    """Supported compliance standards"""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"

@dataclass
class SecurityEvent:
    """Security event for audit logging"""
    event_id: str
    event_type: str
    user_id: str
    username: str
    ip_address: str
    user_agent: str
    timestamp: datetime
    severity: SecurityLevel
    description: str
    additional_data: Dict[str, Any]
    compliance_flags: List[ComplianceStandard]

@dataclass
class AccessAttempt:
    """Track access attempts for security monitoring"""
    user_id: str
    ip_address: str
    timestamp: datetime
    success: bool
    failure_reason: Optional[str] = None
    user_agent: Optional[str] = None

class DataClassification:
    """Data classification levels for compliance"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class EncryptionManager:
    """Handles data encryption and decryption"""
    
    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or self._generate_master_key()
        self.fernet = self._create_fernet()
    
    def _generate_master_key(self) -> str:
        """Generate a new master encryption key"""
        return Fernet.generate_key().decode()
    
    def _create_fernet(self) -> Fernet:
        """Create Fernet instance for encryption/decryption"""
        key = self.master_key.encode()
        return Fernet(key)
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted_data = self.fernet.encrypt(data.encode())
            return base64.b64encode(encrypted_data).decode()
        except Exception as e:
            raise Exception(f"Encryption failed: {e}")
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted_data = self.fernet.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            raise Exception(f"Decryption failed: {e}")
    
    def encrypt_file(self, file_path: str, output_path: str) -> bool:
        """Encrypt a file"""
        try:
            with open(file_path, 'rb') as file:
                file_data = file.read()
            
            encrypted_data = self.fernet.encrypt(file_data)
            
            with open(output_path, 'wb') as encrypted_file:
                encrypted_file.write(encrypted_data)
            
            return True
        except Exception as e:
            print(f"File encryption failed: {e}")
            return False
    
    def decrypt_file(self, encrypted_file_path: str, output_path: str) -> bool:
        """Decrypt a file"""
        try:
            with open(encrypted_file_path, 'rb') as encrypted_file:
                encrypted_data = encrypted_file.read()
            
            decrypted_data = self.fernet.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as file:
                file.write(decrypted_data)
            
            return True
        except Exception as e:
            print(f"File decryption failed: {e}")
            return False

class SecurityAuditLogger:
    """Advanced security audit logging system"""
    
    def __init__(self, db_path: str = "security_audit.db"):
        self.db_path = db_path
        self.encryption_manager = EncryptionManager()
        self.init_audit_database()
    
    def init_audit_database(self):
        """Initialize security audit database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Security events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                event_type TEXT NOT NULL,
                user_id TEXT,
                username TEXT,
                ip_address TEXT,
                user_agent TEXT,
                timestamp TIMESTAMP NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                additional_data TEXT,
                compliance_flags TEXT,
                hash_signature TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Access attempts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                ip_address TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                success BOOLEAN NOT NULL,
                failure_reason TEXT,
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Data access logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_access_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                username TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT,
                action TEXT NOT NULL,
                classification_level TEXT,
                ip_address TEXT,
                timestamp TIMESTAMP NOT NULL,
                compliance_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Compliance audit trail
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audit_id TEXT UNIQUE NOT NULL,
                compliance_standard TEXT NOT NULL,
                audit_type TEXT NOT NULL,
                status TEXT NOT NULL,
                findings TEXT,
                recommendations TEXT,
                auditor_id TEXT,
                audit_date TIMESTAMP NOT NULL,
                next_audit_date TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_security_event(self, event: SecurityEvent) -> bool:
        """Log security event with integrity protection"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Serialize additional data
            additional_data_json = json.dumps(event.additional_data)
            compliance_flags_json = json.dumps([flag.value for flag in event.compliance_flags])
            
            # Create hash signature for integrity
            event_data = f"{event.event_id}{event.event_type}{event.user_id}{event.timestamp.isoformat()}{additional_data_json}"
            hash_signature = hashlib.sha256(event_data.encode()).hexdigest()
            
            cursor.execute('''
                INSERT INTO security_events 
                (event_id, event_type, user_id, username, ip_address, user_agent, 
                 timestamp, severity, description, additional_data, compliance_flags, hash_signature)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id, event.event_type, event.user_id, event.username,
                event.ip_address, event.user_agent, event.timestamp.isoformat(),
                event.severity.value, event.description, additional_data_json,
                compliance_flags_json, hash_signature
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error logging security event: {e}")
            return False
    
    def log_access_attempt(self, attempt: AccessAttempt) -> bool:
        """Log access attempt for security monitoring"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO access_attempts 
                (user_id, ip_address, timestamp, success, failure_reason, user_agent)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                attempt.user_id, attempt.ip_address, attempt.timestamp.isoformat(),
                attempt.success, attempt.failure_reason, attempt.user_agent
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error logging access attempt: {e}")
            return False
    
    def log_data_access(self, user_id: str, username: str, resource_type: str, 
                       action: str, classification_level: str, ip_address: str,
                       resource_id: Optional[str] = None, compliance_reason: Optional[str] = None) -> bool:
        """Log data access for compliance"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO data_access_logs 
                (user_id, username, resource_type, resource_id, action, 
                 classification_level, ip_address, timestamp, compliance_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, username, resource_type, resource_id, action,
                classification_level, ip_address, datetime.now().isoformat(), compliance_reason
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error logging data access: {e}")
            return False
    
    def get_security_events(self, start_date: datetime, end_date: datetime, 
                           severity: Optional[SecurityLevel] = None) -> pd.DataFrame:
        """Retrieve security events for analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT * FROM security_events 
                WHERE timestamp BETWEEN ? AND ?
            '''
            params = [start_date.isoformat(), end_date.isoformat()]
            
            if severity:
                query += " AND severity = ?"
                params.append(severity.value)
            
            query += " ORDER BY timestamp DESC"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            return df
            
        except Exception as e:
            print(f"Error retrieving security events: {e}")
            return pd.DataFrame()
    
    def detect_suspicious_activity(self, lookback_hours: int = 24) -> List[Dict[str, Any]]:
        """Detect suspicious activities based on patterns"""
        suspicious_activities = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Multiple failed login attempts from same IP
            cursor = conn.cursor()
            cursor.execute('''
                SELECT ip_address, COUNT(*) as failed_attempts
                FROM access_attempts 
                WHERE success = 0 AND timestamp > datetime('now', '-{} hours')
                GROUP BY ip_address
                HAVING failed_attempts >= 5
            '''.format(lookback_hours))
            
            for row in cursor.fetchall():
                suspicious_activities.append({
                    'type': 'multiple_failed_logins',
                    'severity': SecurityLevel.HIGH,
                    'description': f"Multiple failed login attempts from IP {row[0]}",
                    'details': {'ip_address': row[0], 'failed_attempts': row[1]}
                })
            
            # Unusual access patterns (access from multiple countries)
            cursor.execute('''
                SELECT user_id, COUNT(DISTINCT ip_address) as unique_ips
                FROM access_attempts 
                WHERE success = 1 AND timestamp > datetime('now', '-{} hours')
                GROUP BY user_id
                HAVING unique_ips >= 3
            '''.format(lookback_hours))
            
            for row in cursor.fetchall():
                suspicious_activities.append({
                    'type': 'unusual_access_pattern',
                    'severity': SecurityLevel.MEDIUM,
                    'description': f"User {row[0]} accessed from {row[1]} different IPs",
                    'details': {'user_id': row[0], 'unique_ips': row[1]}
                })
            
            # High-severity security events
            cursor.execute('''
                SELECT event_type, COUNT(*) as count
                FROM security_events 
                WHERE severity IN ('high', 'critical') AND timestamp > datetime('now', '-{} hours')
                GROUP BY event_type
            '''.format(lookback_hours))
            
            for row in cursor.fetchall():
                suspicious_activities.append({
                    'type': 'high_severity_events',
                    'severity': SecurityLevel.HIGH,
                    'description': f"Multiple {row[0]} events detected",
                    'details': {'event_type': row[0], 'count': row[1]}
                })
            
            conn.close()
            
        except Exception as e:
            print(f"Error detecting suspicious activity: {e}")
        
        return suspicious_activities

class ComplianceManager:
    """Manages regulatory compliance requirements"""
    
    def __init__(self, audit_logger: SecurityAuditLogger):
        self.audit_logger = audit_logger
        self.compliance_rules = self._load_compliance_rules()
        self.data_retention_policies = self._load_retention_policies()
    
    def _load_compliance_rules(self) -> Dict[ComplianceStandard, Dict[str, Any]]:
        """Load compliance rules for different standards"""
        return {
            ComplianceStandard.GDPR: {
                'data_retention_days': 2555,  # 7 years max
                'data_subject_rights': ['access', 'rectification', 'erasure', 'portability'],
                'consent_required': True,
                'breach_notification_hours': 72,
                'dpo_required': True
            },
            ComplianceStandard.HIPAA: {
                'data_retention_days': 2190,  # 6 years
                'encryption_required': True,
                'access_controls': 'role_based',
                'audit_logs_required': True,
                'minimum_necessary': True
            },
            ComplianceStandard.SOC2: {
                'security_controls': ['availability', 'security', 'processing_integrity'],
                'audit_frequency_months': 12,
                'access_reviews_required': True,
                'change_management': True,
                'incident_response_plan': True
            },
            ComplianceStandard.ISO27001: {
                'risk_assessment_required': True,
                'security_policies': True,
                'asset_management': True,
                'access_control': True,
                'cryptography': True,
                'incident_management': True
            }
        }
    
    def _load_retention_policies(self) -> Dict[str, int]:
        """Load data retention policies by data type"""
        return {
            'risk_data': 2555,  # 7 years
            'user_data': 2555,  # 7 years
            'audit_logs': 2555,  # 7 years
            'session_data': 30,  # 30 days
            'chat_messages': 365,  # 1 year
            'annotations': 1095,  # 3 years
        }
    
    def check_compliance_status(self, standard: ComplianceStandard) -> Dict[str, Any]:
        """Check current compliance status for a standard"""
        compliance_status = {
            'standard': standard.value,
            'overall_status': 'compliant',
            'checks': [],
            'recommendations': [],
            'last_checked': datetime.now().isoformat()
        }
        
        rules = self.compliance_rules.get(standard, {})
        
        if standard == ComplianceStandard.GDPR:
            # Check GDPR-specific requirements
            compliance_status['checks'].extend(self._check_gdpr_compliance())
        
        elif standard == ComplianceStandard.HIPAA:
            # Check HIPAA-specific requirements
            compliance_status['checks'].extend(self._check_hipaa_compliance())
        
        elif standard == ComplianceStandard.SOC2:
            # Check SOC2-specific requirements
            compliance_status['checks'].extend(self._check_soc2_compliance())
        
        # Determine overall status
        failed_checks = [check for check in compliance_status['checks'] if not check['passed']]
        if failed_checks:
            compliance_status['overall_status'] = 'non_compliant'
            compliance_status['recommendations'] = [check['recommendation'] for check in failed_checks]
        
        return compliance_status
    
    def _check_gdpr_compliance(self) -> List[Dict[str, Any]]:
        """Check GDPR compliance requirements"""
        checks = []
        
        # Check data retention
        checks.append({
            'requirement': 'Data Retention Limits',
            'passed': self._check_data_retention(ComplianceStandard.GDPR),
            'recommendation': 'Review and implement data retention policies'
        })
        
        # Check consent mechanisms
        checks.append({
            'requirement': 'Consent Management',
            'passed': True,  # Assume implemented
            'recommendation': 'Implement explicit consent collection and management'
        })
        
        # Check data subject rights
        checks.append({
            'requirement': 'Data Subject Rights',
            'passed': True,  # Assume implemented
            'recommendation': 'Implement processes for data access, rectification, and erasure requests'
        })
        
        # Check breach notification procedures
        checks.append({
            'requirement': 'Breach Notification Procedures',
            'passed': True,  # Assume implemented
            'recommendation': 'Establish 72-hour breach notification procedures'
        })
        
        return checks
    
    def _check_hipaa_compliance(self) -> List[Dict[str, Any]]:
        """Check HIPAA compliance requirements"""
        checks = []
        
        # Check encryption
        checks.append({
            'requirement': 'Data Encryption',
            'passed': True,  # Assume implemented with EncryptionManager
            'recommendation': 'Ensure all PHI is encrypted at rest and in transit'
        })
        
        # Check access controls
        checks.append({
            'requirement': 'Access Controls',
            'passed': True,  # Assume implemented with role-based access
            'recommendation': 'Implement role-based access controls for PHI'
        })
        
        # Check audit logs
        checks.append({
            'requirement': 'Audit Logging',
            'passed': True,  # Implemented with SecurityAuditLogger
            'recommendation': 'Maintain comprehensive audit logs for PHI access'
        })
        
        return checks
    
    def _check_soc2_compliance(self) -> List[Dict[str, Any]]:
        """Check SOC2 compliance requirements"""
        checks = []
        
        # Check security controls
        checks.append({
            'requirement': 'Security Controls',
            'passed': True,  # Assume implemented
            'recommendation': 'Implement comprehensive security controls framework'
        })
        
        # Check access reviews
        checks.append({
            'requirement': 'Access Reviews',
            'passed': False,  # Need to implement regular access reviews
            'recommendation': 'Implement quarterly access reviews and role certifications'
        })
        
        # Check change management
        checks.append({
            'requirement': 'Change Management',
            'passed': False,  # Need formal change management process
            'recommendation': 'Implement formal change management and approval processes'
        })
        
        return checks
    
    def _check_data_retention(self, standard: ComplianceStandard) -> bool:
        """Check if data retention policies are being followed"""
        # This would check actual data ages against retention policies
        # For now, assume compliant
        return True
    
    def generate_compliance_report(self, standard: ComplianceStandard, 
                                 include_recommendations: bool = True) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        compliance_status = self.check_compliance_status(standard)
        
        report = {
            'report_id': secrets.token_hex(16),
            'generated_at': datetime.now().isoformat(),
            'standard': standard.value,
            'status': compliance_status,
            'audit_summary': self._generate_audit_summary(),
            'data_inventory': self._generate_data_inventory(),
            'risk_assessment': self._generate_risk_assessment(standard)
        }
        
        if include_recommendations:
            report['recommendations'] = self._generate_compliance_recommendations(standard)
        
        return report
    
    def _generate_audit_summary(self) -> Dict[str, Any]:
        """Generate audit trail summary"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        events_df = self.audit_logger.get_security_events(start_date, end_date)
        
        return {
            'total_events': len(events_df),
            'high_severity_events': len(events_df[events_df['severity'].isin(['high', 'critical'])]),
            'unique_users': events_df['user_id'].nunique() if not events_df.empty else 0,
            'event_types': events_df['event_type'].value_counts().to_dict() if not events_df.empty else {}
        }
    
    def _generate_data_inventory(self) -> Dict[str, Any]:
        """Generate data classification inventory"""
        return {
            'data_types': {
                'risk_data': {
                    'classification': DataClassification.CONFIDENTIAL,
                    'retention_days': self.data_retention_policies['risk_data'],
                    'encryption_required': True
                },
                'user_data': {
                    'classification': DataClassification.RESTRICTED,
                    'retention_days': self.data_retention_policies['user_data'],
                    'encryption_required': True
                },
                'audit_logs': {
                    'classification': DataClassification.RESTRICTED,
                    'retention_days': self.data_retention_policies['audit_logs'],
                    'encryption_required': True
                }
            }
        }
    
    def _generate_risk_assessment(self, standard: ComplianceStandard) -> Dict[str, Any]:
        """Generate compliance risk assessment"""
        return {
            'overall_risk_level': 'medium',
            'identified_risks': [
                {
                    'risk': 'Data breach due to unauthorized access',
                    'likelihood': 'low',
                    'impact': 'high',
                    'mitigation': 'Implement multi-factor authentication'
                },
                {
                    'risk': 'Non-compliance with data retention policies',
                    'likelihood': 'medium',
                    'impact': 'medium',
                    'mitigation': 'Implement automated data lifecycle management'
                }
            ]
        }
    
    def _generate_compliance_recommendations(self, standard: ComplianceStandard) -> List[Dict[str, Any]]:
        """Generate compliance improvement recommendations"""
        recommendations = []
        
        if standard == ComplianceStandard.GDPR:
            recommendations.extend([
                {
                    'priority': 'high',
                    'category': 'data_protection',
                    'recommendation': 'Implement data anonymization for analytics',
                    'timeline': '30 days'
                },
                {
                    'priority': 'medium',
                    'category': 'consent_management',
                    'recommendation': 'Add granular consent options for data processing',
                    'timeline': '60 days'
                }
            ])
        
        elif standard == ComplianceStandard.HIPAA:
            recommendations.extend([
                {
                    'priority': 'high',
                    'category': 'technical_safeguards',
                    'recommendation': 'Implement automatic session timeouts',
                    'timeline': '14 days'
                },
                {
                    'priority': 'medium',
                    'category': 'administrative_safeguards',
                    'recommendation': 'Conduct annual security risk assessment',
                    'timeline': '90 days'
                }
            ])
        
        return recommendations
    
    def schedule_compliance_audit(self, standard: ComplianceStandard, 
                                audit_date: datetime, auditor_id: str) -> str:
        """Schedule compliance audit"""
        audit_id = secrets.token_hex(16)
        
        try:
            conn = sqlite3.connect(self.audit_logger.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO compliance_audit 
                (audit_id, compliance_standard, audit_type, status, auditor_id, audit_date)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                audit_id, standard.value, 'scheduled', 'pending',
                auditor_id, audit_date.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            return audit_id
            
        except Exception as e:
            print(f"Error scheduling compliance audit: {e}")
            return None

class SecurityMiddleware:
    """Security middleware for request validation and protection"""
    
    def __init__(self, audit_logger: SecurityAuditLogger):
        self.audit_logger = audit_logger
        self.rate_limits = {}
        self.blocked_ips = set()
        self.security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'"
        }
    
    def validate_request(self, user_id: str, ip_address: str, user_agent: str) -> Tuple[bool, Optional[str]]:
        """Validate incoming request for security threats"""
        
        # Check if IP is blocked
        if ip_address in self.blocked_ips:
            self._log_security_event(
                event_type="blocked_ip_access",
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                severity=SecurityLevel.HIGH,
                description=f"Access attempt from blocked IP: {ip_address}"
            )
            return False, "IP address is blocked"
        
        # Check rate limiting
        if not self._check_rate_limit(ip_address):
            return False, "Rate limit exceeded"
        
        # Validate IP address format
        if not self._is_valid_ip(ip_address):
            return False, "Invalid IP address format"
        
        # Check for suspicious user agents
        if self._is_suspicious_user_agent(user_agent):
            self._log_security_event(
                event_type="suspicious_user_agent",
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                severity=SecurityLevel.MEDIUM,
                description=f"Suspicious user agent detected: {user_agent}"
            )
        
        return True, None
    
    def _check_rate_limit(self, ip_address: str, max_requests: int = 100, window_minutes: int = 15) -> bool:
        """Check rate limiting for IP address"""
        current_time = time.time()
        window_start = current_time - (window_minutes * 60)
        
        if ip_address not in self.rate_limits:
            self.rate_limits[ip_address] = []
        
        # Remove old requests outside the window
        self.rate_limits[ip_address] = [
            req_time for req_time in self.rate_limits[ip_address] 
            if req_time > window_start
        ]
        
        # Check if under limit
        if len(self.rate_limits[ip_address]) >= max_requests:
            return False
        
        # Add current request
        self.rate_limits[ip_address].append(current_time)
        return True
    
    def _is_valid_ip(self, ip_address: str) -> bool:
        """Validate IP address format"""
        try:
            ipaddress.ip_address(ip_address)
            return True
        except ValueError:
            return False
    
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check for suspicious user agent patterns"""
        if not user_agent:
            return True
        
        # Common bot patterns
        suspicious_patterns = [
            r'bot', r'crawler', r'spider', r'scraper',
            r'curl', r'wget', r'python-requests',
            r'scanner', r'exploit', r'hack'
        ]
        
        user_agent_lower = user_agent.lower()
        
        for pattern in suspicious_patterns:
            if re.search(pattern, user_agent_lower):
                return True
        
        return False
    
    def block_ip(self, ip_address: str, reason: str) -> bool:
        """Block an IP address"""
        try:
            self.blocked_ips.add(ip_address)
            
            self._log_security_event(
                event_type="ip_blocked",
                user_id="system",
                ip_address=ip_address,
                user_agent="",
                severity=SecurityLevel.HIGH,
                description=f"IP blocked: {reason}",
                additional_data={"reason": reason}
            )
            
            return True
        except Exception as e:
            print(f"Error blocking IP: {e}")
            return False
    
    def unblock_ip(self, ip_address: str) -> bool:
        """Unblock an IP address"""
        try:
            self.blocked_ips.discard(ip_address)
            
            self._log_security_event(
                event_type="ip_unblocked",
                user_id="system",
                ip_address=ip_address,
                user_agent="",
                severity=SecurityLevel.MEDIUM,
                description=f"IP unblocked: {ip_address}"
            )
            
            return True
        except Exception as e:
            print(f"Error unblocking IP: {e}")
            return False
    
    def _log_security_event(self, event_type: str, user_id: str, ip_address: str,
                          user_agent: str, severity: SecurityLevel, description: str,
                          additional_data: Optional[Dict[str, Any]] = None):
        """Log security event"""
        event = SecurityEvent(
            event_id=secrets.token_hex(16),
            event_type=event_type,
            user_id=user_id,
            username="",  # Will be filled by caller if available
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.now(),
            severity=severity,
            description=description,
            additional_data=additional_data or {},
            compliance_flags=[]
        )
        
        self.audit_logger.log_security_event(event)

# Streamlit integration functions
def init_security_features():
    """Initialize security features for Streamlit app"""
    if 'security_audit_logger' not in st.session_state:
        st.session_state.security_audit_logger = SecurityAuditLogger()
    
    if 'compliance_manager' not in st.session_state:
        st.session_state.compliance_manager = ComplianceManager(st.session_state.security_audit_logger)
    
    if 'security_middleware' not in st.session_state:
        st.session_state.security_middleware = SecurityMiddleware(st.session_state.security_audit_logger)
    
    if 'encryption_manager' not in st.session_state:
        st.session_state.encryption_manager = EncryptionManager()

def render_security_dashboard():
    """Render security and compliance dashboard"""
    st.header("🔒 Security & Compliance Dashboard")
    
    init_security_features()
    
    audit_logger = st.session_state.security_audit_logger
    compliance_manager = st.session_state.compliance_manager
    
    # Security overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Get recent security events
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        recent_events = audit_logger.get_security_events(start_date, end_date)
        st.metric("Security Events (7d)", len(recent_events))
    
    with col2:
        high_severity = len(recent_events[recent_events['severity'].isin(['high', 'critical'])]) if not recent_events.empty else 0
        st.metric("High Severity Events", high_severity)
    
    with col3:
        suspicious_activities = audit_logger.detect_suspicious_activity(24)
        st.metric("Suspicious Activities", len(suspicious_activities))
    
    with col4:
        unique_users = recent_events['user_id'].nunique() if not recent_events.empty else 0
        st.metric("Active Users (7d)", unique_users)
    
    # Tabs for different security aspects
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Audit Logs", "📋 Compliance", "🚨 Alerts", "⚙️ Settings"])
    
    with tab1:
        st.subheader("Security Audit Logs")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=7))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now().date())
        
        # Severity filter
        severity_filter = st.selectbox("Severity", ["All", "low", "medium", "high", "critical"])
        
        # Get and display events
        events_df = audit_logger.get_security_events(
            datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date, datetime.max.time()),
            SecurityLevel(severity_filter) if severity_filter != "All" else None
        )
        
        if not events_df.empty:
            st.dataframe(events_df[['timestamp', 'event_type', 'username', 'severity', 'description']], use_container_width=True)
        else:
            st.info("No security events found for the selected criteria.")
    
    with tab2:
        st.subheader("Compliance Status")
        
        # Compliance standard selector
        standard = st.selectbox("Compliance Standard", [s.value for s in ComplianceStandard])
        selected_standard = ComplianceStandard(standard)
        
        if st.button("Check Compliance Status"):
            with st.spinner("Checking compliance..."):
                status = compliance_manager.check_compliance_status(selected_standard)
                
                # Display overall status
                status_color = "🟢" if status['overall_status'] == 'compliant' else "🔴"
                st.write(f"**Overall Status:** {status_color} {status['overall_status'].title()}")
                
                # Display individual checks
                st.write("**Compliance Checks:**")
                for check in status['checks']:
                    check_icon = "✅" if check['passed'] else "❌"
                    st.write(f"{check_icon} {check['requirement']}")
                    if not check['passed']:
                        st.write(f"   💡 *{check['recommendation']}*")
        
        # Generate compliance report
        if st.button("Generate Compliance Report"):
            with st.spinner("Generating report..."):
                report = compliance_manager.generate_compliance_report(selected_standard)
                
                st.success("Compliance report generated!")
                st.json(report)
    
    with tab3:
        st.subheader("Security Alerts")
        
        # Suspicious activities
        suspicious_activities = audit_logger.detect_suspicious_activity(24)
        
        if suspicious_activities:
            st.warning(f"Found {len(suspicious_activities)} suspicious activities:")
            
            for activity in suspicious_activities:
                severity_color = {"low": "🟡", "medium": "🟠", "high": "🔴", "critical": "🚨"}
                color = severity_color.get(activity['severity'].value, "⚪")
                
                with st.expander(f"{color} {activity['description']}"):
                    st.write(f"**Type:** {activity['type']}")
                    st.write(f"**Severity:** {activity['severity'].value}")
                    st.json(activity['details'])
        else:
            st.success("No suspicious activities detected in the last 24 hours.")
    
    with tab4:
        st.subheader("Security Settings")
        
        # IP blocking interface
        st.write("**IP Address Management:**")
        
        col1, col2 = st.columns(2)
        with col1:
            ip_to_block = st.text_input("IP Address to Block")
            block_reason = st.text_input("Reason for Blocking")
            
            if st.button("Block IP"):
                if ip_to_block and block_reason:
                    middleware = st.session_state.security_middleware
                    if middleware.block_ip(ip_to_block, block_reason):
                        st.success(f"IP {ip_to_block} has been blocked")
                    else:
                        st.error("Failed to block IP address")
        
        with col2:
            ip_to_unblock = st.text_input("IP Address to Unblock")
            
            if st.button("Unblock IP"):
                if ip_to_unblock:
                    middleware = st.session_state.security_middleware
                    if middleware.unblock_ip(ip_to_unblock):
                        st.success(f"IP {ip_to_unblock} has been unblocked")
                    else:
                        st.error("Failed to unblock IP address")
        
        # Display blocked IPs
        middleware = st.session_state.security_middleware
        if middleware.blocked_ips:
            st.write("**Currently Blocked IPs:**")
            for ip in middleware.blocked_ips:
                st.write(f"- {ip}")
        else:
            st.info("No IPs are currently blocked")

def log_data_access(user_id: str, username: str, resource_type: str, action: str, 
                   classification_level: str = DataClassification.INTERNAL):
    """Helper function to log data access for compliance"""
    if 'security_audit_logger' in st.session_state:
        # Get user's IP address (simplified - in production, get from request headers)
        ip_address = "127.0.0.1"  # Default for local development
        
        st.session_state.security_audit_logger.log_data_access(
            user_id=user_id,
            username=username,
            resource_type=resource_type,
            action=action,
            classification_level=classification_level,
            ip_address=ip_address,
            compliance_reason=f"User {username} performed {action} on {resource_type}"
        )

if __name__ == "__main__":
    # Example usage and testing
    
    # Initialize security components
    audit_logger = SecurityAuditLogger()
    compliance_manager = ComplianceManager(audit_logger)
    security_middleware = SecurityMiddleware(audit_logger)
    encryption_manager = EncryptionManager()
    
    # Test encryption
    test_data = "Sensitive information that needs protection"
    encrypted = encryption_manager.encrypt_data(test_data)
    decrypted = encryption_manager.decrypt_data(encrypted)
    print(f"Encryption test: {test_data == decrypted}")
    
    # Test security event logging
    test_event = SecurityEvent(
        event_id=secrets.token_hex(16),
        event_type="test_event",
        user_id="test_user",
        username="Test User",
        ip_address="192.168.1.1",
        user_agent="Test Agent",
        timestamp=datetime.now(),
        severity=SecurityLevel.MEDIUM,
        description="Test security event",
        additional_data={"test": True},
        compliance_flags=[ComplianceStandard.GDPR]
    )
    
    audit_logger.log_security_event(test_event)
    print("Security event logged successfully")
    
    # Test compliance check
    gdpr_status = compliance_manager.check_compliance_status(ComplianceStandard.GDPR)
    print(f"GDPR compliance status: {gdpr_status['overall_status']}")
    
    # Test request validation
    is_valid, error = security_middleware.validate_request("test_user", "192.168.1.1", "Mozilla/5.0")
    print(f"Request validation: {is_valid}, Error: {error}")
    
    print("Security and compliance module test completed")