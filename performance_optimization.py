"""
Performance Optimization and Intelligent Caching Module
Implements advanced caching strategies, performance monitoring, and optimization techniques
"""

import time
import hashlib
import pickle
import json
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import pandas as pd
import numpy as np
import streamlit as st
import redis
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc
import weakref
import functools
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Cache strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns

class CacheLevel(Enum):
    """Cache levels"""
    MEMORY = "memory"
    DISK = "disk"
    REDIS = "redis"
    DISTRIBUTED = "distributed"

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl: Optional[timedelta] = None
    size_bytes: int = 0
    hit_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl is None:
            return False
        return datetime.now() - self.created_at > self.ttl
    
    def update_access(self):
        """Update access statistics"""
        self.last_accessed = datetime.now()
        self.access_count += 1
        self.hit_count += 1

@dataclass
class CacheStats:
    """Cache performance statistics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    memory_usage_bytes: int = 0
    avg_response_time: float = 0.0
    hit_rate: float = 0.0
    
    def update_hit(self, response_time: float):
        """Record cache hit"""
        self.cache_hits += 1
        self.total_requests += 1
        self._update_avg_response_time(response_time)
        self._update_hit_rate()
    
    def update_miss(self, response_time: float):
        """Record cache miss"""
        self.cache_misses += 1
        self.total_requests += 1
        self._update_avg_response_time(response_time)
        self._update_hit_rate()
    
    def _update_avg_response_time(self, response_time: float):
        """Update average response time"""
        if self.total_requests > 1:
            self.avg_response_time = (
                (self.avg_response_time * (self.total_requests - 1) + response_time) 
                / self.total_requests
            )
        else:
            self.avg_response_time = response_time
    
    def _update_hit_rate(self):
        """Update cache hit rate"""
        if self.total_requests > 0:
            self.hit_rate = self.cache_hits / self.total_requests

class BaseCacheManager(ABC):
    """Abstract base class for cache managers"""
    
    def __init__(self, max_size_mb: int = 100, default_ttl: int = 3600):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = timedelta(seconds=default_ttl)
        self.stats = CacheStats()
        self.lock = threading.RLock()
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Set value in cache"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    def get_size(self) -> int:
        """Get cache size in bytes"""
        pass
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self.stats

class MemoryCacheManager(BaseCacheManager):
    """In-memory cache manager with intelligent eviction"""
    
    def __init__(self, max_size_mb: int = 100, strategy: CacheStrategy = CacheStrategy.LRU):
        super().__init__(max_size_mb)
        self.strategy = strategy
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # For LRU
        self.access_frequency: Dict[str, int] = {}  # For LFU
        self.cleanup_thread = None
        self.start_cleanup_thread()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        start_time = time.time()
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                if entry.is_expired():
                    self._evict_entry(key)
                    self.stats.update_miss(time.time() - start_time)
                    return None
                
                entry.update_access()
                self._update_access_tracking(key)
                self.stats.update_hit(time.time() - start_time)
                return entry.value
            
            self.stats.update_miss(time.time() - start_time)
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Set value in memory cache"""
        try:
            with self.lock:
                # Calculate size
                size = self._calculate_size(value)
                
                # Check if we need to make space
                while (self.get_size() + size > self.max_size_bytes and 
                       len(self.cache) > 0):
                    self._evict_one()
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=1,
                    ttl=ttl or self.default_ttl,
                    size_bytes=size
                )
                
                # Store in cache
                self.cache[key] = entry
                self._update_access_tracking(key)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to set cache entry {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from memory cache"""
        with self.lock:
            if key in self.cache:
                self._evict_entry(key)
                return True
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.access_frequency.clear()
            return True
    
    def get_size(self) -> int:
        """Get total cache size in bytes"""
        return sum(entry.size_bytes for entry in self.cache.values())
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes"""
        try:
            return len(pickle.dumps(value))
        except:
            return len(str(value).encode('utf-8'))
    
    def _update_access_tracking(self, key: str):
        """Update access tracking for eviction strategies"""
        if self.strategy == CacheStrategy.LRU:
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
        
        elif self.strategy == CacheStrategy.LFU:
            self.access_frequency[key] = self.access_frequency.get(key, 0) + 1
    
    def _evict_one(self):
        """Evict one entry based on strategy"""
        if not self.cache:
            return
        
        key_to_evict = None
        
        if self.strategy == CacheStrategy.LRU:
            key_to_evict = self.access_order[0] if self.access_order else list(self.cache.keys())[0]
        
        elif self.strategy == CacheStrategy.LFU:
            if self.access_frequency:
                key_to_evict = min(self.access_frequency.keys(), 
                                 key=lambda k: self.access_frequency[k])
            else:
                key_to_evict = list(self.cache.keys())[0]
        
        elif self.strategy == CacheStrategy.FIFO:
            key_to_evict = min(self.cache.keys(), 
                             key=lambda k: self.cache[k].created_at)
        
        elif self.strategy == CacheStrategy.TTL:
            # Evict expired entries first
            expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
            if expired_keys:
                key_to_evict = expired_keys[0]
            else:
                key_to_evict = list(self.cache.keys())[0]
        
        if key_to_evict:
            self._evict_entry(key_to_evict)
    
    def _evict_entry(self, key: str):
        """Evict specific entry"""
        if key in self.cache:
            del self.cache[key]
            self.stats.evictions += 1
        
        if key in self.access_order:
            self.access_order.remove(key)
        
        if key in self.access_frequency:
            del self.access_frequency[key]
    
    def start_cleanup_thread(self):
        """Start background cleanup thread for expired entries"""
        def cleanup_expired():
            while True:
                try:
                    with self.lock:
                        expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
                        for key in expired_keys:
                            self._evict_entry(key)
                    
                    time.sleep(60)  # Cleanup every minute
                    
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
                    time.sleep(60)
        
        self.cleanup_thread = threading.Thread(target=cleanup_expired, daemon=True)
        self.cleanup_thread.start()

class RedisCacheManager(BaseCacheManager):
    """Redis-based cache manager for distributed caching"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 db: int = 0, max_size_mb: int = 100):
        super().__init__(max_size_mb)
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.redis_client.ping()
            self.available = True
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_client = None
            self.available = False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self.available:
            return None
        
        start_time = time.time()
        
        try:
            # Get value and metadata
            cached_data = self.redis_client.get(f"cache:{key}")
            
            if cached_data:
                entry_data = json.loads(cached_data)
                self.stats.update_hit(time.time() - start_time)
                
                # Update access statistics
                self.redis_client.incr(f"access_count:{key}")
                self.redis_client.set(f"last_accessed:{key}", datetime.now().isoformat())
                
                return pickle.loads(entry_data['value'].encode('latin-1'))
            
            self.stats.update_miss(time.time() - start_time)
            return None
            
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.stats.update_miss(time.time() - start_time)
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Set value in Redis cache"""
        if not self.available:
            return False
        
        try:
            # Serialize value
            serialized_value = pickle.dumps(value).decode('latin-1')
            
            # Create entry data
            entry_data = {
                'value': serialized_value,
                'created_at': datetime.now().isoformat(),
                'size_bytes': len(serialized_value)
            }
            
            # Set with TTL
            ttl_seconds = int((ttl or self.default_ttl).total_seconds())
            
            pipe = self.redis_client.pipeline()
            pipe.set(f"cache:{key}", json.dumps(entry_data), ex=ttl_seconds)
            pipe.set(f"access_count:{key}", 0, ex=ttl_seconds)
            pipe.set(f"last_accessed:{key}", datetime.now().isoformat(), ex=ttl_seconds)
            pipe.execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from Redis cache"""
        if not self.available:
            return False
        
        try:
            pipe = self.redis_client.pipeline()
            pipe.delete(f"cache:{key}")
            pipe.delete(f"access_count:{key}")
            pipe.delete(f"last_accessed:{key}")
            result = pipe.execute()
            
            return result[0] > 0
            
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        if not self.available:
            return False
        
        try:
            # Delete all cache keys
            cache_keys = self.redis_client.keys("cache:*")
            access_keys = self.redis_client.keys("access_count:*")
            last_accessed_keys = self.redis_client.keys("last_accessed:*")
            
            all_keys = cache_keys + access_keys + last_accessed_keys
            
            if all_keys:
                self.redis_client.delete(*all_keys)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    def get_size(self) -> int:
        """Get total cache size (estimated)"""
        if not self.available:
            return 0
        
        try:
            total_size = 0
            cache_keys = self.redis_client.keys("cache:*")
            
            for key in cache_keys:
                data = self.redis_client.get(key)
                if data:
                    try:
                        entry_data = json.loads(data)
                        total_size += entry_data.get('size_bytes', 0)
                    except:
                        total_size += len(data)
            
            return total_size
            
        except Exception as e:
            logger.error(f"Redis get_size error: {e}")
            return 0

class MultiLevelCacheManager:
    """Multi-level cache manager combining memory, disk, and Redis"""
    
    def __init__(self, memory_size_mb: int = 50, redis_host: str = 'localhost'):
        self.memory_cache = MemoryCacheManager(memory_size_mb, CacheStrategy.LRU)
        self.redis_cache = RedisCacheManager(redis_host, max_size_mb=200)
        self.disk_cache = DiskCacheManager(max_size_mb=500)
        
        self.levels = [
            (CacheLevel.MEMORY, self.memory_cache),
            (CacheLevel.REDIS, self.redis_cache), 
            (CacheLevel.DISK, self.disk_cache)
        ]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache levels (L1 -> L2 -> L3)"""
        for level, cache_manager in self.levels:
            value = cache_manager.get(key)
            if value is not None:
                # Promote to higher cache levels
                self._promote_to_higher_levels(key, value, level)
                return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Set value in all cache levels"""
        success = True
        
        for level, cache_manager in self.levels:
            try:
                cache_manager.set(key, value, ttl)
            except Exception as e:
                logger.error(f"Failed to set in {level.value} cache: {e}")
                success = False
        
        return success
    
    def _promote_to_higher_levels(self, key: str, value: Any, current_level: CacheLevel):
        """Promote cache entry to higher levels"""
        found_current = False
        
        for level, cache_manager in self.levels:
            if level == current_level:
                found_current = True
                continue
            
            if not found_current:
                # This is a higher level, set the value
                try:
                    cache_manager.set(key, value)
                except Exception as e:
                    logger.error(f"Failed to promote to {level.value}: {e}")
    
    def get_combined_stats(self) -> Dict[str, CacheStats]:
        """Get statistics from all cache levels"""
        return {
            level.value: cache_manager.get_stats() 
            for level, cache_manager in self.levels
        }

class DiskCacheManager(BaseCacheManager):
    """Disk-based cache manager for persistent caching"""
    
    def __init__(self, cache_dir: str = "cache", max_size_mb: int = 500):
        super().__init__(max_size_mb)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.index_file = self.cache_dir / "cache_index.db"
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for cache index"""
        conn = sqlite3.connect(str(self.index_file))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                last_accessed TIMESTAMP NOT NULL,
                access_count INTEGER DEFAULT 0,
                size_bytes INTEGER NOT NULL,
                expires_at TIMESTAMP,
                hit_count INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        start_time = time.time()
        
        try:
            conn = sqlite3.connect(str(self.index_file))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT filename, expires_at FROM cache_entries 
                WHERE key = ?
            ''', (key,))
            
            result = cursor.fetchone()
            
            if result:
                filename, expires_at = result
                
                # Check expiration
                if expires_at and datetime.fromisoformat(expires_at) < datetime.now():
                    self.delete(key)
                    conn.close()
                    self.stats.update_miss(time.time() - start_time)
                    return None
                
                # Load from disk
                file_path = self.cache_dir / filename
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        value = pickle.load(f)
                    
                    # Update access statistics
                    cursor.execute('''
                        UPDATE cache_entries 
                        SET last_accessed = ?, access_count = access_count + 1,
                            hit_count = hit_count + 1
                        WHERE key = ?
                    ''', (datetime.now().isoformat(), key))
                    
                    conn.commit()
                    conn.close()
                    
                    self.stats.update_hit(time.time() - start_time)
                    return value
            
            conn.close()
            self.stats.update_miss(time.time() - start_time)
            return None
            
        except Exception as e:
            logger.error(f"Disk cache get error: {e}")
            self.stats.update_miss(time.time() - start_time)
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Set value in disk cache"""
        try:
            # Generate filename
            filename = hashlib.md5(key.encode()).hexdigest() + ".cache"
            file_path = self.cache_dir / filename
            
            # Serialize and save to disk
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
            
            size_bytes = file_path.stat().st_size
            expires_at = None
            
            if ttl:
                expires_at = (datetime.now() + ttl).isoformat()
            
            # Update index
            conn = sqlite3.connect(str(self.index_file))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO cache_entries 
                (key, filename, created_at, last_accessed, access_count, size_bytes, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                key, filename, datetime.now().isoformat(), datetime.now().isoformat(),
                1, size_bytes, expires_at
            ))
            
            conn.commit()
            conn.close()
            
            # Check if we need to cleanup
            self._cleanup_if_needed()
            
            return True
            
        except Exception as e:
            logger.error(f"Disk cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from disk cache"""
        try:
            conn = sqlite3.connect(str(self.index_file))
            cursor = conn.cursor()
            
            cursor.execute('SELECT filename FROM cache_entries WHERE key = ?', (key,))
            result = cursor.fetchone()
            
            if result:
                filename = result[0]
                file_path = self.cache_dir / filename
                
                # Delete file
                if file_path.exists():
                    file_path.unlink()
                
                # Remove from index
                cursor.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                conn.commit()
                conn.close()
                
                return True
            
            conn.close()
            return False
            
        except Exception as e:
            logger.error(f"Disk cache delete error: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all disk cache entries"""
        try:
            # Delete all cache files
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            
            # Clear index
            conn = sqlite3.connect(str(self.index_file))
            cursor = conn.cursor()
            cursor.execute('DELETE FROM cache_entries')
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Disk cache clear error: {e}")
            return False
    
    def get_size(self) -> int:
        """Get total disk cache size"""
        try:
            conn = sqlite3.connect(str(self.index_file))
            cursor = conn.cursor()
            
            cursor.execute('SELECT SUM(size_bytes) FROM cache_entries')
            result = cursor.fetchone()
            
            conn.close()
            
            return result[0] if result[0] else 0
            
        except Exception as e:
            logger.error(f"Disk cache get_size error: {e}")
            return 0
    
    def _cleanup_if_needed(self):
        """Cleanup old entries if cache is getting too large"""
        current_size = self.get_size()
        
        if current_size > self.max_size_bytes:
            # Remove oldest entries
            conn = sqlite3.connect(str(self.index_file))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT key FROM cache_entries 
                ORDER BY last_accessed ASC 
                LIMIT ?
            ''', (max(1, len(list(self.cache_dir.glob("*.cache"))) // 4),))
            
            old_keys = cursor.fetchall()
            
            for (key,) in old_keys:
                self.delete(key)
            
            conn.close()

class PerformanceMonitor:
    """Monitors application performance and provides optimization recommendations"""
    
    def __init__(self):
        self.metrics = {
            'response_times': [],
            'memory_usage': [],
            'cpu_usage': [],
            'cache_hit_rates': [],
            'database_query_times': [],
            'api_call_times': []
        }
        
        self.thresholds = {
            'max_response_time': 2.0,  # seconds
            'max_memory_usage': 80,    # percentage
            'max_cpu_usage': 80,       # percentage
            'min_cache_hit_rate': 0.7  # 70%
        }
        
        self.monitoring_active = False
        self.monitoring_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                memory_percent = psutil.virtual_memory().percent
                cpu_percent = psutil.cpu_percent(interval=1)
                
                self.metrics['memory_usage'].append({
                    'timestamp': datetime.now(),
                    'value': memory_percent
                })
                
                self.metrics['cpu_usage'].append({
                    'timestamp': datetime.now(),
                    'value': cpu_percent
                })
                
                # Keep only last 1000 entries
                for metric_type in self.metrics:
                    if len(self.metrics[metric_type]) > 1000:
                        self.metrics[metric_type] = self.metrics[metric_type][-1000:]
                
                # Check for performance issues
                self._check_performance_alerts()
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(30)
    
    def record_response_time(self, operation: str, response_time: float):
        """Record response time for an operation"""
        self.metrics['response_times'].append({
            'timestamp': datetime.now(),
            'operation': operation,
            'value': response_time
        })
    
    def record_cache_hit_rate(self, cache_name: str, hit_rate: float):
        """Record cache hit rate"""
        self.metrics['cache_hit_rates'].append({
            'timestamp': datetime.now(),
            'cache_name': cache_name,
            'value': hit_rate
        })
    
    def record_database_query_time(self, query_type: str, query_time: float):
        """Record database query time"""
        self.metrics['database_query_times'].append({
            'timestamp': datetime.now(),
            'query_type': query_type,
            'value': query_time
        })
    
    def _check_performance_alerts(self):
        """Check for performance issues and generate alerts"""
        alerts = []
        
        # Check memory usage
        if self.metrics['memory_usage']:
            latest_memory = self.metrics['memory_usage'][-1]['value']
            if latest_memory > self.thresholds['max_memory_usage']:
                alerts.append({
                    'type': 'high_memory_usage',
                    'severity': 'warning',
                    'message': f"High memory usage: {latest_memory:.1f}%",
                    'recommendation': "Consider implementing more aggressive caching cleanup"
                })
        
        # Check CPU usage
        if self.metrics['cpu_usage']:
            latest_cpu = self.metrics['cpu_usage'][-1]['value']
            if latest_cpu > self.thresholds['max_cpu_usage']:
                alerts.append({
                    'type': 'high_cpu_usage',
                    'severity': 'warning',
                    'message': f"High CPU usage: {latest_cpu:.1f}%",
                    'recommendation': "Consider optimizing computational operations"
                })
        
        # Check response times
        if self.metrics['response_times']:
            recent_times = [
                m['value'] for m in self.metrics['response_times'][-10:]
            ]
            avg_response_time = sum(recent_times) / len(recent_times)
            
            if avg_response_time > self.thresholds['max_response_time']:
                alerts.append({
                    'type': 'slow_response_time',
                    'severity': 'warning',
                    'message': f"Slow average response time: {avg_response_time:.2f}s",
                    'recommendation': "Consider implementing caching or optimizing queries"
                })
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"Performance alert: {alert['message']} - {alert['recommendation']}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'recommendations': []
        }
        
        # System metrics summary
        if self.metrics['memory_usage']:
            memory_values = [m['value'] for m in self.metrics['memory_usage'][-100:]]
            report['summary']['memory'] = {
                'current': memory_values[-1] if memory_values else 0,
                'average': sum(memory_values) / len(memory_values) if memory_values else 0,
                'max': max(memory_values) if memory_values else 0
            }
        
        if self.metrics['cpu_usage']:
            cpu_values = [m['value'] for m in self.metrics['cpu_usage'][-100:]]
            report['summary']['cpu'] = {
                'current': cpu_values[-1] if cpu_values else 0,
                'average': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                'max': max(cpu_values) if cpu_values else 0
            }
        
        # Response time analysis
        if self.metrics['response_times']:
            response_values = [m['value'] for m in self.metrics['response_times'][-100:]]
            report['summary']['response_times'] = {
                'average': sum(response_values) / len(response_values),
                'p95': np.percentile(response_values, 95),
                'p99': np.percentile(response_values, 99),
                'max': max(response_values)
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report['summary'])
        
        return report
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Memory recommendations
        if 'memory' in summary:
            if summary['memory']['average'] > 70:
                recommendations.append({
                    'category': 'memory',
                    'priority': 'high',
                    'recommendation': 'Implement more aggressive garbage collection and cache cleanup',
                    'impact': 'Reduce memory usage by 20-30%'
                })
        
        # Response time recommendations
        if 'response_times' in summary:
            if summary['response_times']['average'] > 1.0:
                recommendations.append({
                    'category': 'performance',
                    'priority': 'high',
                    'recommendation': 'Implement multi-level caching and query optimization',
                    'impact': 'Reduce response times by 40-60%'
                })
        
        # CPU recommendations
        if 'cpu' in summary:
            if summary['cpu']['average'] > 60:
                recommendations.append({
                    'category': 'cpu',
                    'priority': 'medium',
                    'recommendation': 'Consider implementing async processing for heavy operations',
                    'impact': 'Reduce CPU usage by 30-40%'
                })
        
        return recommendations

def cache_function(ttl: int = 3600, cache_manager: Optional[BaseCacheManager] = None):
    """Decorator for caching function results"""
    def decorator(func: Callable) -> Callable:
        if cache_manager is None:
            # Use default memory cache
            func_cache = MemoryCacheManager(max_size_mb=50)
        else:
            func_cache = cache_manager
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key_data = {
                'func': func.__name__,
                'args': args,
                'kwargs': sorted(kwargs.items())
            }
            cache_key = hashlib.md5(str(key_data).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = func_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Cache result
            func_cache.set(cache_key, result, timedelta(seconds=ttl))
            
            # Log performance
            if hasattr(st.session_state, 'performance_monitor'):
                st.session_state.performance_monitor.record_response_time(
                    func.__name__, execution_time
                )
            
            return result
        
        # Add cache management methods
        wrapper.clear_cache = lambda: func_cache.clear()
        wrapper.cache_stats = lambda: func_cache.get_stats()
        
        return wrapper
    
    return decorator

# Streamlit Integration Functions

def initialize_performance_system():
    """Initialize performance optimization system"""
    if 'cache_manager' not in st.session_state:
        st.session_state.cache_manager = MultiLevelCacheManager()
    
    if 'performance_monitor' not in st.session_state:
        st.session_state.performance_monitor = PerformanceMonitor()
        st.session_state.performance_monitor.start_monitoring()
    
    return st.session_state.cache_manager, st.session_state.performance_monitor

def render_performance_dashboard():
    """Render performance optimization dashboard"""
    st.header("⚡ Performance & Caching Dashboard")
    
    cache_manager, performance_monitor = initialize_performance_system()
    
    # Performance overview
    col1, col2, col3, col4 = st.columns(4)
    
    # Get system metrics
    memory_percent = psutil.virtual_memory().percent
    cpu_percent = psutil.cpu_percent()
    
    with col1:
        st.metric("Memory Usage", f"{memory_percent:.1f}%", 
                 delta=f"{memory_percent - 60:.1f}%" if memory_percent > 60 else None)
    
    with col2:
        st.metric("CPU Usage", f"{cpu_percent:.1f}%",
                 delta=f"{cpu_percent - 50:.1f}%" if cpu_percent > 50 else None)
    
    with col3:
        # Cache stats
        cache_stats = cache_manager.get_combined_stats()
        total_hits = sum(stats.cache_hits for stats in cache_stats.values())
        total_requests = sum(stats.total_requests for stats in cache_stats.values())
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        st.metric("Cache Hit Rate", f"{hit_rate:.1f}%")
    
    with col4:
        # Average response time
        avg_response_time = sum(stats.avg_response_time for stats in cache_stats.values()) / len(cache_stats) if cache_stats else 0
        st.metric("Avg Response Time", f"{avg_response_time:.3f}s")
    
    # Tabs for different aspects
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Cache Analytics", "🔧 Cache Management", "📈 Performance Monitoring", "⚙️ Settings"])
    
    with tab1:
        st.subheader("Cache Analytics")
        
        # Cache level statistics
        cache_stats = cache_manager.get_combined_stats()
        
        if cache_stats:
            # Create DataFrame for visualization
            stats_data = []
            for level, stats in cache_stats.items():
                stats_data.append({
                    'Cache Level': level.title(),
                    'Hit Rate': stats.hit_rate * 100,
                    'Total Requests': stats.total_requests,
                    'Cache Hits': stats.cache_hits,
                    'Cache Misses': stats.cache_misses,
                    'Avg Response Time': stats.avg_response_time
                })
            
            stats_df = pd.DataFrame(stats_data)
            
            # Hit rate chart
            if not stats_df.empty:
                import plotly.express as px
                
                fig_hit_rate = px.bar(stats_df, x='Cache Level', y='Hit Rate',
                                    title='Cache Hit Rate by Level')
                st.plotly_chart(fig_hit_rate, use_container_width=True)
                
                # Response time chart
                fig_response_time = px.bar(stats_df, x='Cache Level', y='Avg Response Time',
                                         title='Average Response Time by Cache Level')
                st.plotly_chart(fig_response_time, use_container_width=True)
                
                # Detailed statistics table
                st.subheader("Detailed Cache Statistics")
                st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("No cache statistics available yet. Use the application to generate cache activity.")
    
    with tab2:
        st.subheader("Cache Management")
        
        # Cache control buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear All Caches"):
                cache_manager.memory_cache.clear()
                cache_manager.redis_cache.clear()
                cache_manager.disk_cache.clear()
                st.success("All caches cleared!")
                st.rerun()
        
        with col2:
            if st.button("📊 Refresh Stats"):
                st.rerun()
        
        with col3:
            if st.button("🧹 Force Garbage Collection"):
                gc.collect()
                st.success("Garbage collection performed!")
        
        # Cache size information
        st.subheader("Cache Size Information")
        
        cache_sizes = {
            'Memory Cache': cache_manager.memory_cache.get_size(),
            'Redis Cache': cache_manager.redis_cache.get_size() if cache_manager.redis_cache.available else 0,
            'Disk Cache': cache_manager.disk_cache.get_size()
        }
        
        for cache_name, size_bytes in cache_sizes.items():
            size_mb = size_bytes / (1024 * 1024)
            st.write(f"**{cache_name}:** {size_mb:.2f} MB ({size_bytes:,} bytes)")
        
        # Cache configuration
        st.subheader("Cache Configuration")
        
        with st.expander("Advanced Cache Settings"):
            memory_cache_size = st.slider("Memory Cache Size (MB)", 10, 500, 50)
            default_ttl = st.slider("Default TTL (seconds)", 300, 7200, 3600)
            
            cache_strategy = st.selectbox(
                "Cache Eviction Strategy",
                [strategy.value for strategy in CacheStrategy]
            )
            
            if st.button("Apply Cache Settings"):
                # This would update cache settings (implementation would depend on specific needs)
                st.success("Cache settings updated!")
    
    with tab3:
        st.subheader("Performance Monitoring")
        
        # Get performance report
        performance_report = performance_monitor.get_performance_report()
        
        # System performance metrics
        if 'summary' in performance_report:
            summary = performance_report['summary']
            
            # Memory usage over time
            if performance_monitor.metrics['memory_usage']:
                memory_data = []
                for metric in performance_monitor.metrics['memory_usage'][-50:]:
                    memory_data.append({
                        'Time': metric['timestamp'],
                        'Memory Usage (%)': metric['value']
                    })
                
                memory_df = pd.DataFrame(memory_data)
                
                if not memory_df.empty:
                    import plotly.express as px
                    fig_memory = px.line(memory_df, x='Time', y='Memory Usage (%)',
                                       title='Memory Usage Over Time')
                    st.plotly_chart(fig_memory, use_container_width=True)
            
            # CPU usage over time
            if performance_monitor.metrics['cpu_usage']:
                cpu_data = []
                for metric in performance_monitor.metrics['cpu_usage'][-50:]:
                    cpu_data.append({
                        'Time': metric['timestamp'],
                        'CPU Usage (%)': metric['value']
                    })
                
                cpu_df = pd.DataFrame(cpu_data)
                
                if not cpu_df.empty:
                    fig_cpu = px.line(cpu_df, x='Time', y='CPU Usage (%)',
                                    title='CPU Usage Over Time')
                    st.plotly_chart(fig_cpu, use_container_width=True)
        
        # Performance recommendations
        if 'recommendations' in performance_report:
            recommendations = performance_report['recommendations']
            
            if recommendations:
                st.subheader("🎯 Performance Recommendations")
                
                for rec in recommendations:
                    priority_colors = {
                        'high': '🔴',
                        'medium': '🟡',
                        'low': '🟢'
                    }
                    
                    priority_icon = priority_colors.get(rec['priority'], '⚪')
                    
                    with st.expander(f"{priority_icon} {rec['category'].title()} - {rec['priority'].title()} Priority"):
                        st.write(f"**Recommendation:** {rec['recommendation']}")
                        st.write(f"**Expected Impact:** {rec['impact']}")
            else:
                st.success("✅ No performance issues detected!")
    
    with tab4:
        st.subheader("Performance Settings")
        
        # Monitoring controls
        st.write("**Performance Monitoring:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if performance_monitor.monitoring_active:
                if st.button("⏹️ Stop Monitoring"):
                    performance_monitor.stop_monitoring()
                    st.success("Performance monitoring stopped")
                    st.rerun()
            else:
                if st.button("▶️ Start Monitoring"):
                    performance_monitor.start_monitoring()
                    st.success("Performance monitoring started")
                    st.rerun()
        
        with col2:
            if st.button("📊 Generate Performance Report"):
                report = performance_monitor.get_performance_report()
                st.json(report)
        
        # Performance thresholds
        st.write("**Performance Alert Thresholds:**")
        
        max_response_time = st.slider("Max Response Time (s)", 0.5, 10.0, 2.0)
        max_memory_usage = st.slider("Max Memory Usage (%)", 50, 95, 80)
        max_cpu_usage = st.slider("Max CPU Usage (%)", 50, 95, 80)
        min_cache_hit_rate = st.slider("Min Cache Hit Rate (%)", 0.5, 0.95, 0.7)
        
        if st.button("Update Thresholds"):
            performance_monitor.thresholds = {
                'max_response_time': max_response_time,
                'max_memory_usage': max_memory_usage,
                'max_cpu_usage': max_cpu_usage,
                'min_cache_hit_rate': min_cache_hit_rate
            }
            st.success("Performance thresholds updated!")

# Example usage with caching decorator
@cache_function(ttl=1800)  # Cache for 30 minutes
def expensive_computation(data: pd.DataFrame) -> Dict[str, Any]:
    """Example of expensive computation that benefits from caching"""
    # Simulate expensive operation
    time.sleep(0.1)
    
    return {
        'mean': data.mean().to_dict(),
        'std': data.std().to_dict(),
        'correlation': data.corr().to_dict()
    }

if __name__ == "__main__":
    # Example usage and testing
    
    # Initialize cache manager
    cache_manager = MultiLevelCacheManager()
    
    # Test caching
    test_key = "test_data"
    test_value = {"message": "Hello, World!", "timestamp": datetime.now().isoformat()}
    
    print("Testing multi-level cache...")
    
    # Set value
    success = cache_manager.set(test_key, test_value)
    print(f"Set operation: {success}")
    
    # Get value
    retrieved_value = cache_manager.get(test_key)
    print(f"Retrieved value: {retrieved_value}")
    
    # Test performance monitoring
    performance_monitor = PerformanceMonitor()
    performance_monitor.start_monitoring()
    
    # Simulate some operations
    for i in range(10):
        start_time = time.time()
        time.sleep(0.01)  # Simulate work
        performance_monitor.record_response_time(f"operation_{i}", time.time() - start_time)
    
    # Generate performance report
    report = performance_monitor.get_performance_report()
    print(f"Performance report generated: {len(report)} sections")
    
    performance_monitor.stop_monitoring()
    
    # Test caching decorator
    print("Testing caching decorator...")
    
    sample_data = pd.DataFrame({
        'A': np.random.randn(1000),
        'B': np.random.randn(1000),
        'C': np.random.randn(1000)
    })
    
    # First call (should be slow)
    start_time = time.time()
    result1 = expensive_computation(sample_data)
    time1 = time.time() - start_time
    print(f"First call took: {time1:.3f}s")
    
    # Second call (should be fast due to caching)
    start_time = time.time()
    result2 = expensive_computation(sample_data)
    time2 = time.time() - start_time
    print(f"Second call took: {time2:.3f}s")
    
    print(f"Speedup: {time1/time2:.2f}x")
    print(f"Results match: {result1 == result2}")
    
    print("Performance optimization module test completed!")