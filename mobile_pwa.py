"""
Mobile Responsive Design and Progressive Web App (PWA) Module
Implements mobile-first design patterns and PWA capabilities for enhanced mobile experience
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import streamlit as st
import pandas as pd
from pathlib import Path

@dataclass
class PWAManifest:
    """PWA manifest configuration"""
    name: str
    short_name: str
    description: str
    start_url: str
    display: str = "standalone"
    theme_color: str = "#1f77b4"
    background_color: str = "#ffffff"
    orientation: str = "portrait-primary"
    scope: str = "/"
    icons: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.icons is None:
            self.icons = [
                {
                    "src": "static/icons/icon-192x192.png",
                    "sizes": "192x192",
                    "type": "image/png"
                },
                {
                    "src": "static/icons/icon-512x512.png",
                    "sizes": "512x512",
                    "type": "image/png"
                }
            ]

class ResponsiveDesign:
    """Handles responsive design and mobile optimization"""
    
    def __init__(self):
        self.breakpoints = {
            'mobile': 768,
            'tablet': 1024,
            'desktop': 1200
        }
        
        self.mobile_css = self._generate_mobile_css()
        self.touch_css = self._generate_touch_css()
        
    def _generate_mobile_css(self) -> str:
        """Generate mobile-responsive CSS"""
        return """
        <style>
        /* Mobile-first responsive design */
        @media screen and (max-width: 768px) {
            /* Streamlit app container adjustments */
            .main .block-container {
                padding-top: 1rem !important;
                padding-bottom: 1rem !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                max-width: none !important;
            }
            
            /* Header optimizations */
            .main h1, .main h2, .main h3 {
                font-size: 1.2rem !important;
                line-height: 1.3 !important;
                margin-bottom: 0.5rem !important;
            }
            
            /* Sidebar adjustments for mobile */
            .sidebar .sidebar-content {
                width: 100% !important;
                padding: 0.5rem !important;
            }
            
            /* Metric cards mobile optimization */
            .metric-container {
                margin-bottom: 0.5rem !important;
            }
            
            .metric-container > div {
                padding: 0.5rem !important;
                border-radius: 8px !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            }
            
            /* Button optimizations */
            .stButton > button {
                width: 100% !important;
                height: 44px !important;
                font-size: 16px !important;
                border-radius: 8px !important;
                margin-bottom: 8px !important;
            }
            
            /* Form input optimizations */
            .stTextInput > div > div > input,
            .stSelectbox > div > div > select,
            .stTextArea > div > div > textarea {
                height: 44px !important;
                font-size: 16px !important;
                border-radius: 8px !important;
            }
            
            /* Chart container responsive */
            .stPlotlyChart {
                width: 100% !important;
                height: 300px !important;
            }
            
            /* Dataframe mobile optimization */
            .dataframe {
                font-size: 12px !important;
                overflow-x: auto !important;
            }
            
            .dataframe td, .dataframe th {
                padding: 4px !important;
                white-space: nowrap !important;
            }
            
            /* Tab optimization */
            .stTabs > div > div > div > div {
                font-size: 14px !important;
                padding: 8px 12px !important;
            }
            
            /* Expander optimization */
            .streamlit-expanderHeader {
                font-size: 14px !important;
                padding: 8px !important;
            }
            
            /* Alert and message optimization */
            .stAlert, .stSuccess, .stWarning, .stError, .stInfo {
                font-size: 14px !important;
                padding: 8px !important;
                margin-bottom: 8px !important;
            }
            
            /* Progress bar optimization */
            .stProgress > div > div {
                height: 8px !important;
            }
            
            /* Column layout optimization */
            .row-widget.stRadio > div, 
            .row-widget.stCheckbox > div {
                flex-direction: column !important;
            }
        }
        
        /* Tablet optimizations */
        @media screen and (min-width: 769px) and (max-width: 1024px) {
            .main .block-container {
                padding-left: 2rem !important;
                padding-right: 2rem !important;
            }
            
            .stPlotlyChart {
                height: 400px !important;
            }
        }
        
        /* Touch-friendly styles */
        .touch-friendly {
            -webkit-tap-highlight-color: rgba(0,0,0,0.1);
            touch-action: manipulation;
        }
        
        .touch-button {
            min-height: 44px !important;
            min-width: 44px !important;
            padding: 12px 16px !important;
            border-radius: 8px !important;
            cursor: pointer;
            user-select: none;
            -webkit-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none;
        }
        
        .touch-button:active {
            transform: translateY(1px);
            box-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }
        
        /* Swipe indicators */
        .swipe-container {
            position: relative;
            overflow-x: auto;
            scrollbar-width: none;
            -ms-overflow-style: none;
        }
        
        .swipe-container::-webkit-scrollbar {
            display: none;
        }
        
        .swipe-indicator {
            position: absolute;
            bottom: 8px;
            right: 8px;
            background: rgba(0,0,0,0.5);
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
        }
        
        /* Loading states for mobile */
        .mobile-loader {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 200px;
            font-size: 16px;
        }
        
        .mobile-loader::after {
            content: "";
            width: 20px;
            height: 20px;
            margin-left: 8px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #1f77b4;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Floating action button */
        .fab {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 56px;
            height: 56px;
            border-radius: 50%;
            background: #1f77b4;
            color: white;
            border: none;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            cursor: pointer;
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        }
        
        .fab:hover {
            background: #1565c0;
            transform: scale(1.1);
        }
        
        /* Bottom navigation for mobile */
        .bottom-nav {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: white;
            border-top: 1px solid #e0e0e0;
            padding: 8px 0;
            z-index: 1000;
            display: flex;
            justify-content: space-around;
        }
        
        .bottom-nav-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 8px;
            text-decoration: none;
            color: #666;
            font-size: 12px;
            min-width: 60px;
        }
        
        .bottom-nav-item.active {
            color: #1f77b4;
        }
        
        .bottom-nav-icon {
            font-size: 18px;
            margin-bottom: 4px;
        }
        
        /* Pull to refresh indicator */
        .pull-to-refresh {
            position: absolute;
            top: -60px;
            left: 50%;
            transform: translateX(-50%);
            padding: 12px;
            background: rgba(31, 119, 180, 0.9);
            color: white;
            border-radius: 20px;
            font-size: 14px;
            transition: all 0.3s ease;
        }
        
        .pull-to-refresh.visible {
            top: 20px;
        }
        
        /* Dark mode optimizations for mobile */
        @media (prefers-color-scheme: dark) {
            .bottom-nav {
                background: #1e1e1e;
                border-top-color: #333;
            }
            
            .bottom-nav-item {
                color: #ccc;
            }
            
            .bottom-nav-item.active {
                color: #64b5f6;
            }
        }
        </style>
        """
    
    def _generate_touch_css(self) -> str:
        """Generate touch-optimized CSS"""
        return """
        <style>
        /* Touch gesture support */
        .swipeable {
            touch-action: pan-x pan-y;
            -webkit-overflow-scrolling: touch;
        }
        
        .draggable {
            touch-action: none;
            user-select: none;
            -webkit-user-drag: none;
        }
        
        /* Improved touch targets */
        .touch-target {
            min-height: 44px;
            min-width: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        /* Haptic feedback simulation */
        .haptic-feedback:active {
            animation: hapticPulse 0.1s ease-out;
        }
        
        @keyframes hapticPulse {
            0% { transform: scale(1); }
            50% { transform: scale(0.95); }
            100% { transform: scale(1); }
        }
        </style>
        """
    
    def apply_mobile_styles(self):
        """Apply mobile-responsive styles to Streamlit app"""
        st.markdown(self.mobile_css, unsafe_allow_html=True)
        st.markdown(self.touch_css, unsafe_allow_html=True)
    
    def is_mobile_device(self) -> bool:
        """Detect if user is on mobile device"""
        # This would typically use user agent detection
        # For Streamlit, we'll use viewport width simulation
        return st.session_state.get('is_mobile', False)
    
    def get_device_type(self) -> str:
        """Get device type based on viewport"""
        # Simplified device detection
        if st.session_state.get('viewport_width', 1200) <= self.breakpoints['mobile']:
            return 'mobile'
        elif st.session_state.get('viewport_width', 1200) <= self.breakpoints['tablet']:
            return 'tablet'
        else:
            return 'desktop'
    
    def create_mobile_columns(self, ratios: List[float], device_type: str = None) -> List:
        """Create responsive columns based on device type"""
        if device_type is None:
            device_type = self.get_device_type()
        
        if device_type == 'mobile':
            # Stack columns vertically on mobile
            return [st.container() for _ in ratios]
        else:
            # Use normal columns on larger screens
            return st.columns(ratios)
    
    def mobile_friendly_dataframe(self, df: pd.DataFrame, max_rows: int = 10):
        """Display mobile-friendly dataframe"""
        if self.get_device_type() == 'mobile':
            # Limit columns and rows for mobile
            display_df = df.head(max_rows)
            if len(df.columns) > 4:
                # Show only first few columns with option to expand
                st.write("**Data Preview** (showing first 4 columns)")
                st.dataframe(display_df.iloc[:, :4], use_container_width=True)
                
                with st.expander("Show all columns"):
                    st.dataframe(display_df, use_container_width=True)
            else:
                st.dataframe(display_df, use_container_width=True)
            
            if len(df) > max_rows:
                st.info(f"Showing {max_rows} of {len(df)} total rows")
        else:
            st.dataframe(df, use_container_width=True)

class PWAManager:
    """Manages Progressive Web App functionality"""
    
    def __init__(self, app_name: str = "LLM Risk Visualizer"):
        self.app_name = app_name
        self.manifest = PWAManifest(
            name=app_name,
            short_name="LLM Risk",
            description="Advanced LLM Risk Assessment and Visualization Platform",
            start_url="/",
            theme_color="#1f77b4",
            background_color="#ffffff"
        )
        
        self.service_worker_js = self._generate_service_worker()
        self.offline_data = {}
    
    def _generate_service_worker(self) -> str:
        """Generate service worker JavaScript"""
        return """
        const CACHE_NAME = 'llm-risk-visualizer-v1';
        const urlsToCache = [
            '/',
            '/static/css/main.css',
            '/static/js/main.js',
            '/static/icons/icon-192x192.png',
            '/static/icons/icon-512x512.png'
        ];
        
        // Install event
        self.addEventListener('install', function(event) {
            event.waitUntil(
                caches.open(CACHE_NAME)
                    .then(function(cache) {
                        console.log('Service Worker: Caching files');
                        return cache.addAll(urlsToCache);
                    })
                    .then(function() {
                        console.log('Service Worker: Installed');
                        return self.skipWaiting();
                    })
            );
        });
        
        // Activate event
        self.addEventListener('activate', function(event) {
            event.waitUntil(
                caches.keys().then(function(cacheNames) {
                    return Promise.all(
                        cacheNames.map(function(cacheName) {
                            if (cacheName !== CACHE_NAME) {
                                console.log('Service Worker: Clearing old cache');
                                return caches.delete(cacheName);
                            }
                        })
                    );
                }).then(function() {
                    console.log('Service Worker: Activated');
                    return self.clients.claim();
                })
            );
        });
        
        // Fetch event - Cache First Strategy for static assets, Network First for API calls
        self.addEventListener('fetch', function(event) {
            const requestUrl = new URL(event.request.url);
            
            // Cache first for static assets
            if (requestUrl.pathname.startsWith('/static/') || 
                requestUrl.pathname.endsWith('.css') || 
                requestUrl.pathname.endsWith('.js') ||
                requestUrl.pathname.endsWith('.png') ||
                requestUrl.pathname.endsWith('.jpg')) {
                
                event.respondWith(
                    caches.match(event.request)
                        .then(function(response) {
                            if (response) {
                                return response;
                            }
                            return fetch(event.request);
                        })
                );
            }
            // Network first for API calls and dynamic content
            else if (requestUrl.pathname.startsWith('/api/') || 
                     event.request.method === 'POST') {
                
                event.respondWith(
                    fetch(event.request)
                        .then(function(response) {
                            // Cache successful responses
                            if (response.status === 200) {
                                const responseClone = response.clone();
                                caches.open(CACHE_NAME)
                                    .then(function(cache) {
                                        cache.put(event.request, responseClone);
                                    });
                            }
                            return response;
                        })
                        .catch(function() {
                            // Return cached version if available
                            return caches.match(event.request)
                                .then(function(response) {
                                    if (response) {
                                        return response;
                                    }
                                    // Return offline page for navigation requests
                                    if (event.request.mode === 'navigate') {
                                        return caches.match('/offline.html');
                                    }
                                });
                        })
                );
            }
            // Default: try network first, fallback to cache
            else {
                event.respondWith(
                    fetch(event.request)
                        .catch(function() {
                            return caches.match(event.request);
                        })
                );
            }
        });
        
        // Background sync for offline data
        self.addEventListener('sync', function(event) {
            if (event.tag === 'background-sync') {
                event.waitUntil(syncOfflineData());
            }
        });
        
        // Push notifications
        self.addEventListener('push', function(event) {
            const options = {
                body: event.data ? event.data.text() : 'New update available',
                tag: 'llm-risk-notification',
                icon: '/static/icons/icon-192x192.png',
                badge: '/static/icons/badge-72x72.png',
                vibrate: [100, 50, 100],
                data: {
                    dateOfArrival: Date.now(),
                    primaryKey: 1
                },
                actions: [
                    {
                        action: 'explore',
                        title: 'View Details',
                        icon: '/static/icons/checkmark.png'
                    },
                    {
                        action: 'close',
                        title: 'Close',
                        icon: '/static/icons/xmark.png'
                    }
                ]
            };
            
            event.waitUntil(
                self.registration.showNotification('LLM Risk Visualizer', options)
            );
        });
        
        // Notification click handling
        self.addEventListener('notificationclick', function(event) {
            event.notification.close();
            
            if (event.action === 'explore') {
                event.waitUntil(
                    clients.openWindow('/')
                );
            }
        });
        
        // Offline data synchronization
        async function syncOfflineData() {
            try {
                const cache = await caches.open(CACHE_NAME);
                const offlineData = await cache.match('/offline-data');
                
                if (offlineData) {
                    const data = await offlineData.json();
                    
                    // Sync data with server
                    const response = await fetch('/api/sync-offline-data', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(data)
                    });
                    
                    if (response.ok) {
                        // Clear offline data after successful sync
                        await cache.delete('/offline-data');
                        console.log('Offline data synced successfully');
                    }
                }
            } catch (error) {
                console.error('Failed to sync offline data:', error);
            }
        }
        """
    
    def create_manifest_file(self, output_path: str = "static/manifest.json"):
        """Create PWA manifest file"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            manifest_data = asdict(self.manifest)
            
            with open(output_path, 'w') as f:
                json.dump(manifest_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error creating manifest file: {e}")
            return False
    
    def create_service_worker_file(self, output_path: str = "static/sw.js"):
        """Create service worker file"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(self.service_worker_js)
            
            return True
        except Exception as e:
            print(f"Error creating service worker file: {e}")
            return False
    
    def generate_pwa_html_headers(self) -> str:
        """Generate HTML headers for PWA"""
        return f"""
        <link rel="manifest" href="/static/manifest.json">
        <meta name="theme-color" content="{self.manifest.theme_color}">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="default">
        <meta name="apple-mobile-web-app-title" content="{self.manifest.short_name}">
        <link rel="apple-touch-icon" href="/static/icons/icon-192x192.png">
        <meta name="msapplication-TileColor" content="{self.manifest.theme_color}">
        <meta name="msapplication-TileImage" content="/static/icons/icon-192x192.png">
        
        <script>
        // Register service worker
        if ('serviceWorker' in navigator) {
            window.addEventListener('load', function() {
                navigator.serviceWorker.register('/static/sw.js')
                    .then(function(registration) {
                        console.log('SW registered: ', registration);
                    })
                    .catch(function(registrationError) {
                        console.log('SW registration failed: ', registrationError);
                    });
            });
        }
        
        // PWA install prompt
        let deferredPrompt;
        window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            deferredPrompt = e;
            showInstallPromotion();
        });
        
        function showInstallPromotion() {
            const installBanner = document.createElement('div');
            installBanner.innerHTML = `
                <div style="position: fixed; top: 0; left: 0; right: 0; background: #1f77b4; color: white; padding: 12px; text-align: center; z-index: 10000;">
                    <span>Install {self.manifest.short_name} for better experience</span>
                    <button onclick="installPWA()" style="margin-left: 12px; background: white; color: #1f77b4; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer;">Install</button>
                    <button onclick="dismissInstall()" style="margin-left: 8px; background: none; color: white; border: 1px solid white; padding: 6px 12px; border-radius: 4px; cursor: pointer;">Later</button>
                </div>
            `;
            document.body.appendChild(installBanner);
            
            // Auto-hide after 10 seconds
            setTimeout(() => {
                if (document.body.contains(installBanner)) {
                    document.body.removeChild(installBanner);
                }
            }, 10000);
        }
        
        function installPWA() {
            if (deferredPrompt) {
                deferredPrompt.prompt();
                deferredPrompt.userChoice.then((choiceResult) => {
                    if (choiceResult.outcome === 'accepted') {
                        console.log('User accepted the install prompt');
                    }
                    deferredPrompt = null;
                });
            }
            dismissInstall();
        }
        
        function dismissInstall() {
            const banner = document.querySelector('[style*="position: fixed"][style*="top: 0"]');
            if (banner) {
                document.body.removeChild(banner);
            }
        }
        
        // Handle offline/online status
        window.addEventListener('online', function() {
            showNetworkStatus('Online', '#4caf50');
        });
        
        window.addEventListener('offline', function() {
            showNetworkStatus('Offline', '#f44336');
        });
        
        function showNetworkStatus(status, color) {
            const statusBar = document.createElement('div');
            statusBar.innerHTML = `
                <div style="position: fixed; bottom: 20px; left: 20px; background: ${color}; color: white; padding: 8px 16px; border-radius: 4px; z-index: 10000; box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
                    ${status}
                </div>
            `;
            document.body.appendChild(statusBar);
            
            setTimeout(() => {
                if (document.body.contains(statusBar)) {
                    document.body.removeChild(statusBar);
                }
            }, 3000);
        }
        
        // Touch gesture support
        let touchStartX = 0;
        let touchStartY = 0;
        
        document.addEventListener('touchstart', function(e) {
            touchStartX = e.touches[0].clientX;
            touchStartY = e.touches[0].clientY;
        });
        
        document.addEventListener('touchend', function(e) {
            if (!touchStartX || !touchStartY) {
                return;
            }
            
            const touchEndX = e.changedTouches[0].clientX;
            const touchEndY = e.changedTouches[0].clientY;
            
            const diffX = touchStartX - touchEndX;
            const diffY = touchStartY - touchEndY;
            
            // Swipe detection
            if (Math.abs(diffX) > Math.abs(diffY)) {
                if (Math.abs(diffX) > 50) {
                    if (diffX > 0) {
                        // Swipe left
                        handleSwipeLeft();
                    } else {
                        // Swipe right
                        handleSwipeRight();
                    }
                }
            }
            
            touchStartX = 0;
            touchStartY = 0;
        });
        
        function handleSwipeLeft() {
            // Navigate to next section/tab
            console.log('Swipe left detected');
        }
        
        function handleSwipeRight() {
            // Navigate to previous section/tab
            console.log('Swipe right detected');
        }
        
        // Pull to refresh
        let pullToRefreshEnabled = true;
        let startY = 0;
        let currentY = 0;
        let pullDistance = 0;
        const pullThreshold = 80;
        
        document.addEventListener('touchstart', function(e) {
            if (window.scrollY === 0 && pullToRefreshEnabled) {
                startY = e.touches[0].clientY;
            }
        });
        
        document.addEventListener('touchmove', function(e) {
            if (startY && window.scrollY === 0) {
                currentY = e.touches[0].clientY;
                pullDistance = currentY - startY;
                
                if (pullDistance > 0) {
                    e.preventDefault();
                    
                    const pullIndicator = document.getElementById('pull-to-refresh');
                    if (pullIndicator) {
                        if (pullDistance > pullThreshold) {
                            pullIndicator.textContent = 'Release to refresh';
                            pullIndicator.style.background = '#4caf50';
                        } else {
                            pullIndicator.textContent = 'Pull to refresh';
                            pullIndicator.style.background = 'rgba(31, 119, 180, 0.9)';
                        }
                        pullIndicator.style.transform = `translateX(-50%) translateY(${Math.min(pullDistance - 60, 20)}px)`;
                        pullIndicator.classList.add('visible');
                    }
                }
            }
        });
        
        document.addEventListener('touchend', function(e) {
            if (pullDistance > pullThreshold) {
                // Trigger refresh
                window.location.reload();
            }
            
            const pullIndicator = document.getElementById('pull-to-refresh');
            if (pullIndicator) {
                pullIndicator.classList.remove('visible');
            }
            
            startY = 0;
            currentY = 0;
            pullDistance = 0;
        });
        </script>
        """
    
    def store_offline_data(self, data: Dict[str, Any]):
        """Store data for offline access"""
        self.offline_data.update(data)
    
    def get_offline_data(self, key: str) -> Any:
        """Retrieve offline data"""
        return self.offline_data.get(key)
    
    def setup_push_notifications(self):
        """Setup push notifications (placeholder for future implementation)"""
        # This would integrate with web push services
        pass

class TouchGestureHandler:
    """Handles touch gestures and mobile interactions"""
    
    def __init__(self):
        self.gesture_handlers = {
            'swipe_left': [],
            'swipe_right': [],
            'swipe_up': [],
            'swipe_down': [],
            'pinch': [],
            'tap': [],
            'long_press': []
        }
    
    def register_gesture_handler(self, gesture: str, handler: callable):
        """Register a gesture handler"""
        if gesture in self.gesture_handlers:
            self.gesture_handlers[gesture].append(handler)
    
    def generate_gesture_js(self) -> str:
        """Generate JavaScript for gesture handling"""
        return """
        <script>
        class TouchGestureHandler {
            constructor() {
                this.touchStartTime = 0;
                this.touchStartX = 0;
                this.touchStartY = 0;
                this.touchEndX = 0;
                this.touchEndY = 0;
                this.longPressTimer = null;
                this.longPressThreshold = 500; // ms
                this.swipeThreshold = 50; // px
                
                this.setupEventListeners();
            }
            
            setupEventListeners() {
                document.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: false });
                document.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false });
                document.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: false });
            }
            
            handleTouchStart(e) {
                this.touchStartTime = Date.now();
                this.touchStartX = e.touches[0].clientX;
                this.touchStartY = e.touches[0].clientY;
                
                // Start long press timer
                this.longPressTimer = setTimeout(() => {
                    this.handleLongPress(e);
                }, this.longPressThreshold);
            }
            
            handleTouchMove(e) {
                // Cancel long press if finger moves too much
                const moveThreshold = 10;
                const deltaX = Math.abs(e.touches[0].clientX - this.touchStartX);
                const deltaY = Math.abs(e.touches[0].clientY - this.touchStartY);
                
                if (deltaX > moveThreshold || deltaY > moveThreshold) {
                    clearTimeout(this.longPressTimer);
                }
            }
            
            handleTouchEnd(e) {
                clearTimeout(this.longPressTimer);
                
                this.touchEndX = e.changedTouches[0].clientX;
                this.touchEndY = e.changedTouches[0].clientY;
                
                const touchDuration = Date.now() - this.touchStartTime;
                const deltaX = this.touchStartX - this.touchEndX;
                const deltaY = this.touchStartY - this.touchEndY;
                
                // Determine gesture type
                if (Math.abs(deltaX) > this.swipeThreshold || Math.abs(deltaY) > this.swipeThreshold) {
                    this.handleSwipe(deltaX, deltaY);
                } else if (touchDuration < 300) {
                    this.handleTap(e);
                }
            }
            
            handleSwipe(deltaX, deltaY) {
                if (Math.abs(deltaX) > Math.abs(deltaY)) {
                    // Horizontal swipe
                    if (deltaX > 0) {
                        this.triggerGesture('swipe_left');
                    } else {
                        this.triggerGesture('swipe_right');
                    }
                } else {
                    // Vertical swipe
                    if (deltaY > 0) {
                        this.triggerGesture('swipe_up');
                    } else {
                        this.triggerGesture('swipe_down');
                    }
                }
            }
            
            handleTap(e) {
                this.triggerGesture('tap', { x: this.touchEndX, y: this.touchEndY });
            }
            
            handleLongPress(e) {
                this.triggerGesture('long_press', { x: this.touchStartX, y: this.touchStartY });
            }
            
            triggerGesture(gestureType, data = {}) {
                const event = new CustomEvent('gesture', {
                    detail: { type: gestureType, ...data }
                });
                document.dispatchEvent(event);
            }
        }
        
        // Initialize gesture handler
        const gestureHandler = new TouchGestureHandler();
        
        // Global gesture event listener
        document.addEventListener('gesture', function(e) {
            console.log('Gesture detected:', e.detail);
            
            // Handle specific gestures
            switch(e.detail.type) {
                case 'swipe_left':
                    handleSwipeLeft();
                    break;
                case 'swipe_right':
                    handleSwipeRight();
                    break;
                case 'tap':
                    handleTap(e.detail);
                    break;
                case 'long_press':
                    handleLongPress(e.detail);
                    break;
            }
        });
        
        function handleSwipeLeft() {
            // Navigate to next tab/section
            const tabs = document.querySelectorAll('[data-testid="stTabs"] button');
            const activeTab = document.querySelector('[data-testid="stTabs"] button[aria-selected="true"]');
            
            if (activeTab && tabs.length > 1) {
                const currentIndex = Array.from(tabs).indexOf(activeTab);
                const nextIndex = (currentIndex + 1) % tabs.length;
                tabs[nextIndex].click();
            }
        }
        
        function handleSwipeRight() {
            // Navigate to previous tab/section
            const tabs = document.querySelectorAll('[data-testid="stTabs"] button');
            const activeTab = document.querySelector('[data-testid="stTabs"] button[aria-selected="true"]');
            
            if (activeTab && tabs.length > 1) {
                const currentIndex = Array.from(tabs).indexOf(activeTab);
                const prevIndex = currentIndex === 0 ? tabs.length - 1 : currentIndex - 1;
                tabs[prevIndex].click();
            }
        }
        
        function handleTap(data) {
            // Add ripple effect to tapped elements
            const element = document.elementFromPoint(data.x, data.y);
            if (element) {
                addRippleEffect(element, data.x, data.y);
            }
        }
        
        function handleLongPress(data) {
            // Show context menu or additional options
            const element = document.elementFromPoint(data.x, data.y);
            if (element) {
                showContextualOptions(element, data.x, data.y);
            }
        }
        
        function addRippleEffect(element, x, y) {
            const ripple = document.createElement('div');
            const rect = element.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            
            ripple.style.cssText = `
                position: absolute;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.3);
                pointer-events: none;
                width: ${size}px;
                height: ${size}px;
                left: ${x - rect.left - size/2}px;
                top: ${y - rect.top - size/2}px;
                animation: ripple 0.6s ease-out;
                z-index: 1000;
            `;
            
            element.style.position = 'relative';
            element.appendChild(ripple);
            
            setTimeout(() => {
                if (element.contains(ripple)) {
                    element.removeChild(ripple);
                }
            }, 600);
        }
        
        function showContextualOptions(element, x, y) {
            // Show context menu (placeholder)
            console.log('Long press detected on element:', element);
        }
        
        // Add ripple animation CSS
        const style = document.createElement('style');
        style.textContent = `
            @keyframes ripple {
                from {
                    transform: scale(0);
                    opacity: 1;
                }
                to {
                    transform: scale(1);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
        </script>
        """

class MobileOptimizedComponents:
    """Mobile-optimized UI components"""
    
    def __init__(self, responsive_design: ResponsiveDesign):
        self.responsive_design = responsive_design
    
    def mobile_metric_card(self, title: str, value: str, delta: str = None, 
                          delta_color: str = "normal") -> str:
        """Create mobile-optimized metric card"""
        delta_html = ""
        if delta:
            color_map = {
                "normal": "#666",
                "inverse": "#666", 
                "off": "#666"
            }
            delta_html = f'<div style="font-size: 14px; color: {color_map[delta_color]}; margin-top: 4px;">{delta}</div>'
        
        return f"""
        <div style="
            background: white;
            padding: 16px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 12px;
            border-left: 4px solid #1f77b4;
        ">
            <div style="font-size: 14px; color: #666; margin-bottom: 4px;">{title}</div>
            <div style="font-size: 24px; font-weight: bold; color: #333;">{value}</div>
            {delta_html}
        </div>
        """
    
    def mobile_chart_container(self, chart, title: str = "", height: int = 300):
        """Create mobile-optimized chart container"""
        device_type = self.responsive_design.get_device_type()
        
        if device_type == 'mobile':
            height = min(height, 300)  # Limit height on mobile
        
        st.markdown(f"### {title}" if title else "")
        
        # Add swipe indicator for mobile
        if device_type == 'mobile':
            st.markdown(
                '<div class="swipe-container">'
                '<div class="swipe-indicator">Swipe to explore</div>',
                unsafe_allow_html=True
            )
        
        st.plotly_chart(chart, use_container_width=True, height=height)
        
        if device_type == 'mobile':
            st.markdown('</div>', unsafe_allow_html=True)
    
    def mobile_action_sheet(self, title: str, actions: List[Dict[str, str]]):
        """Create mobile action sheet"""
        st.markdown(f"**{title}**")
        
        for action in actions:
            if st.button(
                action['label'], 
                key=action.get('key', action['label']),
                help=action.get('description', '')
            ):
                if 'callback' in action:
                    action['callback']()
    
    def mobile_bottom_navigation(self, nav_items: List[Dict[str, str]]):
        """Create mobile bottom navigation"""
        nav_html = '<div class="bottom-nav">'
        
        for item in nav_items:
            active_class = "active" if item.get('active', False) else ""
            nav_html += f"""
            <a href="{item['href']}" class="bottom-nav-item {active_class}">
                <div class="bottom-nav-icon">{item['icon']}</div>
                <div>{item['label']}</div>
            </a>
            """
        
        nav_html += '</div>'
        st.markdown(nav_html, unsafe_allow_html=True)
    
    def mobile_pull_to_refresh(self):
        """Add pull-to-refresh indicator"""
        st.markdown(
            '<div id="pull-to-refresh" class="pull-to-refresh">Pull to refresh</div>',
            unsafe_allow_html=True
        )
    
    def mobile_fab(self, icon: str = "➕", action: str = "fab_clicked"):
        """Create floating action button"""
        fab_html = f"""
        <button class="fab" onclick="window.parent.postMessage({{type: '{action}'}}, '*')">
            {icon}
        </button>
        """
        st.markdown(fab_html, unsafe_allow_html=True)

# Streamlit integration functions
def initialize_mobile_pwa():
    """Initialize mobile and PWA features"""
    if 'responsive_design' not in st.session_state:
        st.session_state.responsive_design = ResponsiveDesign()
    
    if 'pwa_manager' not in st.session_state:
        st.session_state.pwa_manager = PWAManager()
    
    if 'touch_handler' not in st.session_state:
        st.session_state.touch_handler = TouchGestureHandler()
    
    if 'mobile_components' not in st.session_state:
        st.session_state.mobile_components = MobileOptimizedComponents(
            st.session_state.responsive_design
        )
    
    # Apply mobile styles
    st.session_state.responsive_design.apply_mobile_styles()
    
    # Add PWA headers
    pwa_headers = st.session_state.pwa_manager.generate_pwa_html_headers()
    st.markdown(pwa_headers, unsafe_allow_html=True)
    
    # Add gesture handling
    gesture_js = st.session_state.touch_handler.generate_gesture_js()
    st.markdown(gesture_js, unsafe_allow_html=True)

def render_mobile_settings():
    """Render mobile and PWA settings"""
    st.header("📱 Mobile & PWA Settings")
    
    initialize_mobile_pwa()
    
    responsive_design = st.session_state.responsive_design
    pwa_manager = st.session_state.pwa_manager
    
    tab1, tab2, tab3 = st.tabs(["📱 Mobile Settings", "🔧 PWA Config", "👆 Touch Settings"])
    
    with tab1:
        st.subheader("Mobile Display Settings")
        
        # Device simulation
        device_type = st.selectbox(
            "Simulate Device Type",
            ["desktop", "tablet", "mobile"],
            index=["desktop", "tablet", "mobile"].index(responsive_design.get_device_type())
        )
        
        if device_type != responsive_design.get_device_type():
            viewport_widths = {"desktop": 1200, "tablet": 800, "mobile": 400}
            st.session_state.viewport_width = viewport_widths[device_type]
            st.rerun()
        
        # Mobile preferences
        st.write("**Mobile Preferences:**")
        
        enable_pull_refresh = st.checkbox("Enable Pull to Refresh", value=True)
        enable_gestures = st.checkbox("Enable Touch Gestures", value=True)
        enable_haptics = st.checkbox("Enable Haptic Feedback", value=True)
        
        compact_mode = st.checkbox("Compact Display Mode", value=device_type == "mobile")
        
        if compact_mode:
            st.info("Compact mode reduces spacing and font sizes for better mobile experience")
    
    with tab2:
        st.subheader("Progressive Web App Configuration")
        
        # PWA settings
        app_name = st.text_input("App Name", value=pwa_manager.manifest.name)
        short_name = st.text_input("Short Name", value=pwa_manager.manifest.short_name)
        description = st.text_area("Description", value=pwa_manager.manifest.description)
        
        theme_color = st.color_picker("Theme Color", value=pwa_manager.manifest.theme_color)
        background_color = st.color_picker("Background Color", value=pwa_manager.manifest.background_color)
        
        # Update manifest
        if st.button("Update PWA Configuration"):
            pwa_manager.manifest.name = app_name
            pwa_manager.manifest.short_name = short_name
            pwa_manager.manifest.description = description
            pwa_manager.manifest.theme_color = theme_color
            pwa_manager.manifest.background_color = background_color
            
            st.success("PWA configuration updated!")
        
        # Generate files
        if st.button("Generate PWA Files"):
            manifest_created = pwa_manager.create_manifest_file()
            sw_created = pwa_manager.create_service_worker_file()
            
            if manifest_created and sw_created:
                st.success("PWA files generated successfully!")
                st.info("Files created: static/manifest.json, static/sw.js")
            else:
                st.error("Failed to generate PWA files")
        
        # Installation stats (placeholder)
        st.write("**Installation Analytics:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Install Prompts", "127")
        with col2:
            st.metric("Installations", "89")
        with col3:
            st.metric("Install Rate", "70.1%")
    
    with tab3:
        st.subheader("Touch Gesture Settings")
        
        # Gesture sensitivity
        swipe_threshold = st.slider("Swipe Sensitivity", 30, 100, 50)
        long_press_duration = st.slider("Long Press Duration (ms)", 300, 1000, 500)
        
        # Gesture actions
        st.write("**Gesture Actions:**")
        
        swipe_left_action = st.selectbox("Swipe Left", ["Next Tab", "Back", "None"])
        swipe_right_action = st.selectbox("Swipe Right", ["Previous Tab", "Forward", "None"])
        long_press_action = st.selectbox("Long Press", ["Context Menu", "Selection", "None"])
        
        # Test gestures
        st.write("**Test Gestures:**")
        
        if st.button("Test Swipe Left"):
            st.success("Swipe left gesture triggered!")
        
        if st.button("Test Swipe Right"):
            st.success("Swipe right gesture triggered!")
        
        if st.button("Test Long Press"):
            st.success("Long press gesture triggered!")
        
        # Gesture statistics
        st.write("**Gesture Usage Statistics:**")
        
        gesture_data = pd.DataFrame({
            'Gesture': ['Swipe Left', 'Swipe Right', 'Tap', 'Long Press', 'Pinch'],
            'Usage Count': [45, 32, 234, 12, 8],
            'Success Rate': [92, 88, 98, 78, 85]
        })
        
        st.dataframe(gesture_data, use_container_width=True)

def render_mobile_dashboard():
    """Render mobile-optimized dashboard"""
    initialize_mobile_pwa()
    
    responsive_design = st.session_state.responsive_design
    mobile_components = st.session_state.mobile_components
    
    # Add pull-to-refresh
    mobile_components.mobile_pull_to_refresh()
    
    st.title("📱 Mobile Dashboard")
    
    if responsive_design.get_device_type() == 'mobile':
        st.info("Optimized for mobile viewing")
    
    # Mobile-optimized metrics
    col1, col2 = responsive_design.create_mobile_columns([1, 1], responsive_design.get_device_type())
    
    with col1:
        st.markdown(
            mobile_components.mobile_metric_card("Active Users", "1,234", "+12%"),
            unsafe_allow_html=True
        )
        st.markdown(
            mobile_components.mobile_metric_card("Risk Events", "89", "-5%"),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            mobile_components.mobile_metric_card("System Health", "98.5%", "+0.2%"),
            unsafe_allow_html=True
        )
        st.markdown(
            mobile_components.mobile_metric_card("Data Sources", "12", "+2"),
            unsafe_allow_html=True
        )
    
    # Mobile action sheet
    mobile_components.mobile_action_sheet(
        "Quick Actions",
        [
            {"label": "🔄 Refresh Data", "key": "refresh", "description": "Reload all data"},
            {"label": "📊 View Reports", "key": "reports", "description": "Open reports page"},
            {"label": "⚙️ Settings", "key": "settings", "description": "App settings"}
        ]
    )
    
    # Add floating action button
    mobile_components.mobile_fab("📊", "quick_analysis")
    
    # Bottom navigation for mobile
    if responsive_design.get_device_type() == 'mobile':
        mobile_components.mobile_bottom_navigation([
            {"href": "#dashboard", "icon": "🏠", "label": "Home", "active": True},
            {"href": "#analytics", "icon": "📊", "label": "Analytics"},
            {"href": "#settings", "icon": "⚙️", "label": "Settings"},
            {"href": "#profile", "icon": "👤", "label": "Profile"}
        ])

if __name__ == "__main__":
    # Example usage and testing
    
    # Initialize components
    responsive_design = ResponsiveDesign()
    pwa_manager = PWAManager("LLM Risk Visualizer Test")
    touch_handler = TouchGestureHandler()
    mobile_components = MobileOptimizedComponents(responsive_design)
    
    # Test PWA file generation
    print("Testing PWA file generation...")
    manifest_created = pwa_manager.create_manifest_file("test_manifest.json")
    sw_created = pwa_manager.create_service_worker_file("test_sw.js")
    
    if manifest_created and sw_created:
        print("✅ PWA files generated successfully")
    else:
        print("❌ Failed to generate PWA files")
    
    # Test responsive design
    print("Testing responsive design...")
    print(f"Device type detection: {responsive_design.get_device_type()}")
    print(f"Is mobile: {responsive_design.is_mobile_device()}")
    
    # Test mobile components
    print("Testing mobile components...")
    metric_card = mobile_components.mobile_metric_card("Test Metric", "123", "+5%")
    print(f"Metric card generated: {len(metric_card)} characters")
    
    # Test offline data storage
    print("Testing offline data storage...")
    pwa_manager.store_offline_data({"test_key": "test_value", "timestamp": datetime.now().isoformat()})
    retrieved_data = pwa_manager.get_offline_data("test_key")
    print(f"Offline data retrieved: {retrieved_data}")
    
    print("Mobile and PWA module test completed successfully!")