"""
Monitoring and Alerting system for LLM Risk Visualizer
Provides real-time monitoring, automated alerts, and system health tracking
"""

import time
import threading
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
import schedule

from database import DatabaseManager
from config import RISK_THRESHOLDS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class AlertRule:
    """Define alert rule configuration"""
    id: str
    name: str
    description: str
    condition: Dict[str, Any]
    severity: str  # 'low', 'medium', 'high', 'critical'
    enabled: bool = True
    cooldown_minutes: int = 60
    last_triggered: Optional[datetime] = None
    notification_methods: List[str] = None

    def __post_init__(self):
        if self.notification_methods is None:
            self.notification_methods = ['email']

@dataclass
class Alert:
    """Represent a triggered alert"""
    rule_id: str
    rule_name: str
    severity: str
    message: str
    details: Dict[str, Any]
    triggered_at: datetime
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

class NotificationManager:
    """Manage different notification channels"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def send_email(self, recipients: List[str], subject: str, body: str, is_html: bool = False):
        """Send email notification"""
        try:
            smtp_config = self.config.get('smtp', {})
            if not smtp_config:
                self.logger.error("SMTP configuration not found")
                return False
            
            msg = MimeMultipart()
            msg['From'] = smtp_config.get('from_email')
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            content_type = 'html' if is_html else 'plain'
            msg.attach(MimeText(body, content_type))
            
            server = smtplib.SMTP(smtp_config.get('server'), smtp_config.get('port', 587))
            if smtp_config.get('use_tls', True):
                server.starttls()
            
            if smtp_config.get('username') and smtp_config.get('password'):
                server.login(smtp_config.get('username'), smtp_config.get('password'))
            
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email sent to {recipients}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False
    
    def send_slack(self, webhook_url: str, message: str, channel: str = None):
        """Send Slack notification"""
        try:
            payload = {
                'text': message,
                'username': 'LLM Risk Monitor',
                'icon_emoji': ':warning:'
            }
            
            if channel:
                payload['channel'] = channel
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            self.logger.info("Slack notification sent")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def send_webhook(self, webhook_url: str, payload: Dict[str, Any]):
        """Send generic webhook notification"""
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Webhook notification sent to {webhook_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook: {e}")
            return False

class RiskMonitor:
    """Monitor risk metrics and detect threshold violations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.logger = logging.getLogger(__name__)
        self.alert_rules = {}
        self.active_alerts = {}
        self.load_default_rules()
    
    def load_default_rules(self):
        """Load default monitoring rules"""
        default_rules = [
            AlertRule(
                id="high_risk_threshold",
                name="High Risk Threshold Exceeded",
                description="Risk rate exceeds high threshold",
                condition={
                    "metric": "risk_rate",
                    "operator": ">",
                    "threshold": RISK_THRESHOLDS['high']
                },
                severity="high",
                cooldown_minutes=30
            ),
            AlertRule(
                id="anomaly_detection",
                name="Risk Anomaly Detected",
                description="Unusual risk pattern detected",
                condition={
                    "metric": "anomaly_score",
                    "operator": ">",
                    "threshold": 0.8
                },
                severity="medium",
                cooldown_minutes=60
            ),
            AlertRule(
                id="data_freshness",
                name="Stale Data Alert",
                description="No new data received recently",
                condition={
                    "metric": "data_age_hours",
                    "operator": ">",
                    "threshold": 24
                },
                severity="medium",
                cooldown_minutes=120
            ),
            AlertRule(
                id="critical_risk_spike",
                name="Critical Risk Spike",
                description="Sudden increase in risk rate",
                condition={
                    "metric": "risk_rate_change",
                    "operator": ">",
                    "threshold": 0.3
                },
                severity="critical",
                cooldown_minutes=15
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.id] = rule
    
    def add_alert_rule(self, rule: AlertRule):
        """Add custom alert rule"""
        self.alert_rules[rule.id] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self.logger.info(f"Removed alert rule: {rule_id}")
    
    def check_risk_thresholds(self) -> List[Alert]:
        """Check for risk threshold violations"""
        alerts = []
        
        try:
            # Get latest risk data
            latest_data = self.db.risk_data.get_latest_data_by_model()
            
            if latest_data.empty:
                return alerts
            
            # Check high risk threshold rule
            high_risk_rule = self.alert_rules.get("high_risk_threshold")
            if high_risk_rule and high_risk_rule.enabled:
                high_risk_data = latest_data[
                    latest_data['risk_rate'] > high_risk_rule.condition['threshold']
                ]
                
                if not high_risk_data.empty and self._can_trigger_alert(high_risk_rule):
                    for _, row in high_risk_data.iterrows():
                        alert = Alert(
                            rule_id=high_risk_rule.id,
                            rule_name=high_risk_rule.name,
                            severity=high_risk_rule.severity,
                            message=f"High risk detected: {row['model']} - {row['language']} - {row['risk_category']}",
                            details={
                                'model': row['model'],
                                'language': row['language'],
                                'risk_category': row['risk_category'],
                                'risk_rate': row['risk_rate'],
                                'threshold': high_risk_rule.condition['threshold']
                            },
                            triggered_at=datetime.now()
                        )
                        alerts.append(alert)
                    
                    high_risk_rule.last_triggered = datetime.now()
            
            # Check for risk spikes
            spike_rule = self.alert_rules.get("critical_risk_spike")
            if spike_rule and spike_rule.enabled:
                risk_changes = self._calculate_risk_changes()
                
                if not risk_changes.empty:
                    spikes = risk_changes[
                        risk_changes['change'] > spike_rule.condition['threshold']
                    ]
                    
                    if not spikes.empty and self._can_trigger_alert(spike_rule):
                        for _, row in spikes.iterrows():
                            alert = Alert(
                                rule_id=spike_rule.id,
                                rule_name=spike_rule.name,
                                severity=spike_rule.severity,
                                message=f"Risk spike detected: {row['model']} - {row['risk_category']}",
                                details={
                                    'model': row['model'],
                                    'language': row['language'],
                                    'risk_category': row['risk_category'],
                                    'previous_rate': row['previous_rate'],
                                    'current_rate': row['current_rate'],
                                    'change': row['change']
                                },
                                triggered_at=datetime.now()
                            )
                            alerts.append(alert)
                        
                        spike_rule.last_triggered = datetime.now()
        
        except Exception as e:
            self.logger.error(f"Error checking risk thresholds: {e}")
        
        return alerts
    
    def check_data_freshness(self) -> List[Alert]:
        """Check for stale data"""
        alerts = []
        
        try:
            freshness_rule = self.alert_rules.get("data_freshness")
            if not freshness_rule or not freshness_rule.enabled:
                return alerts
            
            # Get latest data timestamp
            query = "SELECT MAX(date) as latest_date FROM risk_data"
            result = self.db.connection.execute_query(query)
            
            if not result.empty and result.iloc[0]['latest_date']:
                latest_date = datetime.fromisoformat(result.iloc[0]['latest_date'])
                hours_old = (datetime.now() - latest_date).total_seconds() / 3600
                
                if hours_old > freshness_rule.condition['threshold'] and self._can_trigger_alert(freshness_rule):
                    alert = Alert(
                        rule_id=freshness_rule.id,
                        rule_name=freshness_rule.name,
                        severity=freshness_rule.severity,
                        message=f"Data is {hours_old:.1f} hours old",
                        details={
                            'latest_date': latest_date.isoformat(),
                            'hours_old': hours_old,
                            'threshold': freshness_rule.condition['threshold']
                        },
                        triggered_at=datetime.now()
                    )
                    alerts.append(alert)
                    freshness_rule.last_triggered = datetime.now()
        
        except Exception as e:
            self.logger.error(f"Error checking data freshness: {e}")
        
        return alerts
    
    def check_anomalies(self) -> List[Alert]:
        """Check for detected anomalies"""
        alerts = []
        
        try:
            anomaly_rule = self.alert_rules.get("anomaly_detection")
            if not anomaly_rule or not anomaly_rule.enabled:
                return alerts
            
            # Get recent unacknowledged anomalies
            recent_anomalies = self.db.anomalies.get_anomalies(
                acknowledged=False,
                days=1
            )
            
            if not recent_anomalies.empty and self._can_trigger_alert(anomaly_rule):
                for _, row in recent_anomalies.iterrows():
                    if row['anomaly_score'] > anomaly_rule.condition['threshold']:
                        alert = Alert(
                            rule_id=anomaly_rule.id,
                            rule_name=anomaly_rule.name,
                            severity=anomaly_rule.severity,
                            message=f"Anomaly detected: {row['model']} - {row['risk_category']}",
                            details={
                                'anomaly_id': row['id'],
                                'model': row['model'],
                                'language': row['language'],
                                'risk_category': row['risk_category'],
                                'anomaly_score': row['anomaly_score'],
                                'expected_rate': row['expected_rate'],
                                'actual_rate': row['actual_rate']
                            },
                            triggered_at=datetime.now()
                        )
                        alerts.append(alert)
                
                if alerts:
                    anomaly_rule.last_triggered = datetime.now()
        
        except Exception as e:
            self.logger.error(f"Error checking anomalies: {e}")
        
        return alerts
    
    def _can_trigger_alert(self, rule: AlertRule) -> bool:
        """Check if alert can be triggered based on cooldown"""
        if not rule.last_triggered:
            return True
        
        cooldown_period = timedelta(minutes=rule.cooldown_minutes)
        return datetime.now() - rule.last_triggered >= cooldown_period
    
    def _calculate_risk_changes(self) -> pd.DataFrame:
        """Calculate risk rate changes between recent time periods"""
        try:
            # Get data from last 2 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=2)
            
            data = self.db.risk_data.get_risk_data(start_date=start_date, end_date=end_date)
            
            if data.empty:
                return pd.DataFrame()
            
            # Group by model, language, risk_category and calculate changes
            data['date'] = pd.to_datetime(data['date'])
            data_sorted = data.sort_values(['model', 'language', 'risk_category', 'date'])
            
            changes = []
            for (model, language, category), group in data_sorted.groupby(['model', 'language', 'risk_category']):
                if len(group) >= 2:
                    current_rate = group.iloc[-1]['risk_rate']
                    previous_rate = group.iloc[-2]['risk_rate']
                    change = current_rate - previous_rate
                    
                    changes.append({
                        'model': model,
                        'language': language,
                        'risk_category': category,
                        'current_rate': current_rate,
                        'previous_rate': previous_rate,
                        'change': change
                    })
            
            return pd.DataFrame(changes)
        
        except Exception as e:
            self.logger.error(f"Error calculating risk changes: {e}")
            return pd.DataFrame()

class AlertManager:
    """Manage alerts and notifications"""
    
    def __init__(self, notification_manager: NotificationManager, db_manager: DatabaseManager):
        self.notification_manager = notification_manager
        self.db = db_manager
        self.logger = logging.getLogger(__name__)
        self.alert_history = []
    
    def process_alerts(self, alerts: List[Alert]):
        """Process and send notifications for alerts"""
        for alert in alerts:
            self._send_notifications(alert)
            self._store_alert(alert)
            self.alert_history.append(alert)
            
            self.logger.warning(f"Alert triggered: {alert.rule_name} - {alert.message}")
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        try:
            # Prepare notification content
            subject = f"[{alert.severity.upper()}] LLM Risk Alert: {alert.rule_name}"
            
            message = self._format_alert_message(alert)
            html_message = self._format_alert_html(alert)
            
            # Send email notifications
            email_config = self.notification_manager.config.get('email', {})
            if email_config.get('enabled', False):
                recipients = email_config.get('recipients', [])
                if recipients:
                    self.notification_manager.send_email(
                        recipients, subject, html_message, is_html=True
                    )
            
            # Send Slack notifications
            slack_config = self.notification_manager.config.get('slack', {})
            if slack_config.get('enabled', False):
                webhook_url = slack_config.get('webhook_url')
                if webhook_url:
                    slack_message = f"🚨 *{alert.rule_name}*\\n{alert.message}\\n_Severity: {alert.severity}_"
                    self.notification_manager.send_slack(webhook_url, slack_message)
            
            # Send webhook notifications
            webhook_config = self.notification_manager.config.get('webhook', {})
            if webhook_config.get('enabled', False):
                webhook_url = webhook_config.get('url')
                if webhook_url:
                    payload = {
                        'alert_type': 'risk_monitoring',
                        'rule_id': alert.rule_id,
                        'rule_name': alert.rule_name,
                        'severity': alert.severity,
                        'message': alert.message,
                        'details': alert.details,
                        'triggered_at': alert.triggered_at.isoformat()
                    }
                    self.notification_manager.send_webhook(webhook_url, payload)
        
        except Exception as e:
            self.logger.error(f"Error sending notifications: {e}")
    
    def _format_alert_message(self, alert: Alert) -> str:
        """Format alert message for plain text"""
        message = f"""
LLM Risk Visualizer Alert

Rule: {alert.rule_name}
Severity: {alert.severity.upper()}
Time: {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S')}

Message: {alert.message}

Details:
"""
        
        for key, value in alert.details.items():
            message += f"  {key}: {value}\\n"
        
        return message
    
    def _format_alert_html(self, alert: Alert) -> str:
        """Format alert message for HTML email"""
        severity_colors = {
            'low': '#28a745',
            'medium': '#ffc107',
            'high': '#fd7e14',
            'critical': '#dc3545'
        }
        
        color = severity_colors.get(alert.severity, '#6c757d')
        
        html = f"""
        <html>
        <body>
            <h2 style="color: {color};">🚨 LLM Risk Alert</h2>
            
            <table style="border-collapse: collapse; width: 100%;">
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Rule:</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{alert.rule_name}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Severity:</td>
                    <td style="border: 1px solid #ddd; padding: 8px; color: {color}; font-weight: bold;">
                        {alert.severity.upper()}
                    </td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Time:</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">
                        {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S')}
                    </td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Message:</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{alert.message}</td>
                </tr>
            </table>
            
            <h3>Details:</h3>
            <table style="border-collapse: collapse; width: 100%;">
        """
        
        for key, value in alert.details.items():
            html += f"""
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">{key}:</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{value}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <p style="margin-top: 20px; color: #6c757d; font-size: 12px;">
                This alert was generated by LLM Risk Visualizer monitoring system.
            </p>
        </body>
        </html>
        """
        
        return html
    
    def _store_alert(self, alert: Alert):
        """Store alert in database"""
        try:
            query = '''
                INSERT INTO alerts 
                (rule_id, rule_name, severity, message, details, triggered_at)
                VALUES (?, ?, ?, ?, ?, ?)
            '''
            
            # Create alerts table if it doesn't exist
            create_table_query = '''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id TEXT NOT NULL,
                    rule_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,
                    triggered_at TIMESTAMP NOT NULL,
                    acknowledged BOOLEAN DEFAULT 0,
                    acknowledged_by TEXT,
                    acknowledged_at TIMESTAMP
                )
            '''
            
            self.db.connection.execute_update(create_table_query)
            
            params = (
                alert.rule_id,
                alert.rule_name,
                alert.severity,
                alert.message,
                json.dumps(alert.details),
                alert.triggered_at.isoformat()
            )
            
            self.db.connection.execute_update(query, params)
        
        except Exception as e:
            self.logger.error(f"Error storing alert: {e}")
    
    def acknowledge_alert(self, alert_id: int, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        try:
            query = '''
                UPDATE alerts 
                SET acknowledged = 1, acknowledged_by = ?, acknowledged_at = ?
                WHERE id = ?
            '''
            
            params = (acknowledged_by, datetime.now().isoformat(), alert_id)
            return self.db.connection.execute_update(query, params)
        
        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {e}")
            return False
    
    def get_active_alerts(self) -> pd.DataFrame:
        """Get active (unacknowledged) alerts"""
        try:
            query = '''
                SELECT * FROM alerts 
                WHERE acknowledged = 0 
                ORDER BY triggered_at DESC
            '''
            return self.db.connection.execute_query(query)
        except Exception as e:
            self.logger.error(f"Error fetching active alerts: {e}")
            return pd.DataFrame()

class MonitoringService:
    """Main monitoring service orchestrating all monitoring components"""
    
    def __init__(self, config: Dict[str, Any], db_manager: DatabaseManager):
        self.config = config
        self.db = db_manager
        self.notification_manager = NotificationManager(config.get('notifications', {}))
        self.risk_monitor = RiskMonitor(db_manager)
        self.alert_manager = AlertManager(self.notification_manager, db_manager)
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start the monitoring service"""
        if self.running:
            self.logger.warning("Monitoring service is already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Schedule periodic checks
        schedule.every(5).minutes.do(self._run_monitoring_cycle)
        schedule.every(1).hour.do(self._cleanup_old_alerts)
        
        self.logger.info("Monitoring service started")
    
    def stop_monitoring(self):
        """Stop the monitoring service"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        schedule.clear()
        self.logger.info("Monitoring service stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _run_monitoring_cycle(self):
        """Run a complete monitoring cycle"""
        try:
            self.logger.info("Starting monitoring cycle")
            
            all_alerts = []
            
            # Check risk thresholds
            risk_alerts = self.risk_monitor.check_risk_thresholds()
            all_alerts.extend(risk_alerts)
            
            # Check data freshness
            freshness_alerts = self.risk_monitor.check_data_freshness()
            all_alerts.extend(freshness_alerts)
            
            # Check anomalies
            anomaly_alerts = self.risk_monitor.check_anomalies()
            all_alerts.extend(anomaly_alerts)
            
            # Process alerts
            if all_alerts:
                self.alert_manager.process_alerts(all_alerts)
                self.logger.info(f"Processed {len(all_alerts)} alerts")
            else:
                self.logger.info("No alerts triggered")
        
        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {e}")
    
    def _cleanup_old_alerts(self):
        """Clean up old acknowledged alerts"""
        try:
            # Keep alerts for 30 days
            cutoff_date = datetime.now() - timedelta(days=30)
            
            query = '''
                DELETE FROM alerts 
                WHERE acknowledged = 1 AND acknowledged_at < ?
            '''
            
            self.db.connection.execute_update(query, (cutoff_date.isoformat(),))
            self.logger.info("Cleaned up old alerts")
        
        except Exception as e:
            self.logger.error(f"Error cleaning up alerts: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'running': self.running,
            'total_rules': len(self.risk_monitor.alert_rules),
            'active_rules': sum(1 for rule in self.risk_monitor.alert_rules.values() if rule.enabled),
            'active_alerts': len(self.alert_manager.get_active_alerts()),
            'last_check': datetime.now().isoformat(),
            'database_health': self.db.health_check()
        }
    
    def trigger_manual_check(self) -> Dict[str, Any]:
        """Manually trigger monitoring checks"""
        try:
            self._run_monitoring_cycle()
            return {
                'success': True,
                'message': 'Manual monitoring check completed',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Manual check failed: {e}',
                'timestamp': datetime.now().isoformat()
            }

# Example configuration
MONITORING_CONFIG = {
    'notifications': {
        'smtp': {
            'server': 'smtp.gmail.com',
            'port': 587,
            'use_tls': True,
            'username': 'your-email@gmail.com',
            'password': 'your-app-password',
            'from_email': 'llm-monitor@yourcompany.com'
        },
        'email': {
            'enabled': True,
            'recipients': ['admin@yourcompany.com', 'security@yourcompany.com']
        },
        'slack': {
            'enabled': False,
            'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        },
        'webhook': {
            'enabled': False,
            'url': 'https://your-webhook-endpoint.com/alerts'
        }
    }
}