"""
Internationalization and Multi-Language Support Module
Provides comprehensive i18n support for the LLM Risk Visualizer platform
"""

import json
import os
import re
import locale
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import streamlit as st
import pandas as pd
from babel import Locale, dates, numbers
from babel.support import Format
import gettext

@dataclass
class LanguageConfig:
    """Language configuration"""
    code: str  # ISO 639-1 language code
    name: str  # Language name in English
    native_name: str  # Language name in native language
    flag: str  # Flag emoji
    rtl: bool = False  # Right-to-left text direction
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    number_format: str = "#,##0.##"
    currency_symbol: str = "$"
    decimal_separator: str = "."
    thousands_separator: str = ","

class TranslationManager:
    """Manages translations and localization"""
    
    def __init__(self, translations_dir: str = "translations"):
        self.translations_dir = Path(translations_dir)
        self.translations_dir.mkdir(exist_ok=True)
        
        self.supported_languages = {
            'en': LanguageConfig(
                code='en',
                name='English',
                native_name='English',
                flag='🇺🇸',
                date_format='%m/%d/%Y',
                currency_symbol='$'
            ),
            'zh': LanguageConfig(
                code='zh',
                name='Chinese',
                native_name='中文',
                flag='🇨🇳',
                date_format='%Y年%m月%d日',
                currency_symbol='¥'
            ),
            'es': LanguageConfig(
                code='es',
                name='Spanish',
                native_name='Español',
                flag='🇪🇸',
                date_format='%d/%m/%Y',
                currency_symbol='€'
            ),
            'fr': LanguageConfig(
                code='fr',
                name='French',
                native_name='Français',
                flag='🇫🇷',
                date_format='%d/%m/%Y',
                currency_symbol='€',
                decimal_separator=',',
                thousands_separator=' '
            ),
            'de': LanguageConfig(
                code='de',
                name='German',
                native_name='Deutsch',
                flag='🇩🇪',
                date_format='%d.%m.%Y',
                currency_symbol='€',
                decimal_separator=',',
                thousands_separator='.'
            ),
            'ja': LanguageConfig(
                code='ja',
                name='Japanese',
                native_name='日本語',
                flag='🇯🇵',
                date_format='%Y年%m月%d日',
                currency_symbol='¥'
            ),
            'ko': LanguageConfig(
                code='ko',
                name='Korean',
                native_name='한국어',
                flag='🇰🇷',
                date_format='%Y년 %m월 %d일',
                currency_symbol='₩'
            ),
            'ar': LanguageConfig(
                code='ar',
                name='Arabic',
                native_name='العربية',
                flag='🇸🇦',
                rtl=True,
                date_format='%d/%m/%Y',
                currency_symbol='ر.س'
            ),
            'ru': LanguageConfig(
                code='ru',
                name='Russian',
                native_name='Русский',
                flag='🇷🇺',
                date_format='%d.%m.%Y',
                currency_symbol='₽',
                decimal_separator=',',
                thousands_separator=' '
            ),
            'pt': LanguageConfig(
                code='pt',
                name='Portuguese',
                native_name='Português',
                flag='🇧🇷',
                date_format='%d/%m/%Y',
                currency_symbol='R$',
                decimal_separator=',',
                thousands_separator='.'
            )
        }
        
        self.translations: Dict[str, Dict[str, str]] = {}
        self.current_language = 'en'
        self.fallback_language = 'en'
        
        # Load existing translations
        self.load_all_translations()
        
        # Generate default translations if they don't exist
        self.generate_default_translations()
    
    def load_all_translations(self):
        """Load all translation files"""
        for lang_code in self.supported_languages.keys():
            self.load_translations(lang_code)
    
    def load_translations(self, language_code: str):
        """Load translations for a specific language"""
        translation_file = self.translations_dir / f"{language_code}.json"
        
        if translation_file.exists():
            try:
                with open(translation_file, 'r', encoding='utf-8') as f:
                    self.translations[language_code] = json.load(f)
            except Exception as e:
                print(f"Error loading translations for {language_code}: {e}")
                self.translations[language_code] = {}
        else:
            self.translations[language_code] = {}
    
    def save_translations(self, language_code: str):
        """Save translations for a specific language"""
        translation_file = self.translations_dir / f"{language_code}.json"
        
        try:
            with open(translation_file, 'w', encoding='utf-8') as f:
                json.dump(
                    self.translations.get(language_code, {}), 
                    f, 
                    ensure_ascii=False, 
                    indent=2,
                    sort_keys=True
                )
        except Exception as e:
            print(f"Error saving translations for {language_code}: {e}")
    
    def generate_default_translations(self):
        """Generate default translation strings"""
        default_strings = {
            # Navigation
            'nav.dashboard': 'Dashboard',
            'nav.analytics': 'Analytics',
            'nav.models': 'Models',
            'nav.risks': 'Risks',
            'nav.settings': 'Settings',
            'nav.help': 'Help',
            
            # Dashboard
            'dashboard.title': 'LLM Risk Visualizer',
            'dashboard.welcome': 'Welcome to LLM Risk Visualizer',
            'dashboard.overview': 'Overview',
            'dashboard.recent_activity': 'Recent Activity',
            'dashboard.statistics': 'Statistics',
            
            # Risk Assessment
            'risk.title': 'Risk Assessment',
            'risk.level.low': 'Low Risk',
            'risk.level.medium': 'Medium Risk', 
            'risk.level.high': 'High Risk',
            'risk.level.critical': 'Critical Risk',
            'risk.category.bias': 'Bias',
            'risk.category.toxicity': 'Toxicity',
            'risk.category.misinformation': 'Misinformation',
            'risk.category.privacy': 'Privacy',
            'risk.category.security': 'Security',
            'risk.rate': 'Risk Rate',
            'risk.confidence': 'Confidence',
            'risk.sample_size': 'Sample Size',
            
            # Models
            'model.name': 'Model Name',
            'model.type': 'Model Type',
            'model.version': 'Version',
            'model.language': 'Language',
            'model.performance': 'Performance',
            'model.last_updated': 'Last Updated',
            
            # Analytics
            'analytics.title': 'Analytics',
            'analytics.trends': 'Trends',
            'analytics.comparisons': 'Comparisons',
            'analytics.reports': 'Reports',
            'analytics.export': 'Export',
            
            # Common UI Elements
            'common.save': 'Save',
            'common.cancel': 'Cancel',
            'common.delete': 'Delete',
            'common.edit': 'Edit',
            'common.view': 'View',
            'common.download': 'Download',
            'common.upload': 'Upload',
            'common.search': 'Search',
            'common.filter': 'Filter',
            'common.sort': 'Sort',
            'common.refresh': 'Refresh',
            'common.loading': 'Loading...',
            'common.error': 'Error',
            'common.success': 'Success',
            'common.warning': 'Warning',
            'common.info': 'Information',
            'common.yes': 'Yes',
            'common.no': 'No',
            'common.ok': 'OK',
            'common.apply': 'Apply',
            'common.reset': 'Reset',
            'common.close': 'Close',
            
            # Date and Time
            'date.today': 'Today',
            'date.yesterday': 'Yesterday',
            'date.last_week': 'Last Week',
            'date.last_month': 'Last Month',
            'date.last_year': 'Last Year',
            'time.hours_ago': 'hours ago',
            'time.minutes_ago': 'minutes ago',
            'time.seconds_ago': 'seconds ago',
            
            # Messages
            'message.no_data': 'No data available',
            'message.loading_data': 'Loading data...',
            'message.data_updated': 'Data updated successfully',
            'message.operation_completed': 'Operation completed',
            'message.operation_failed': 'Operation failed',
            'message.invalid_input': 'Invalid input',
            'message.permission_denied': 'Permission denied',
            'message.network_error': 'Network error',
            
            # Settings
            'settings.title': 'Settings',
            'settings.general': 'General',
            'settings.language': 'Language',
            'settings.theme': 'Theme',
            'settings.notifications': 'Notifications',
            'settings.privacy': 'Privacy',
            'settings.security': 'Security',
            'settings.about': 'About',
            
            # Mobile specific
            'mobile.menu': 'Menu',
            'mobile.back': 'Back',
            'mobile.next': 'Next',
            'mobile.previous': 'Previous',
            
            # Collaboration
            'collab.online': 'Online',
            'collab.offline': 'Offline',
            'collab.typing': 'typing...',
            'collab.joined': 'joined the session',
            'collab.left': 'left the session',
            
            # Performance
            'perf.cache_hit_rate': 'Cache Hit Rate',
            'perf.response_time': 'Response Time',
            'perf.memory_usage': 'Memory Usage',
            'perf.cpu_usage': 'CPU Usage',
            
            # Security
            'security.login': 'Login',
            'security.logout': 'Logout',
            'security.username': 'Username',
            'security.password': 'Password',
            'security.access_denied': 'Access Denied',
            'security.session_expired': 'Session Expired'
        }
        
        # Generate translations using placeholder service
        for lang_code in self.supported_languages.keys():
            if lang_code not in self.translations:
                self.translations[lang_code] = {}
            
            # Add missing keys with default values
            for key, default_value in default_strings.items():
                if key not in self.translations[lang_code]:
                    if lang_code == 'en':
                        self.translations[lang_code][key] = default_value
                    else:
                        # Use translation service or keep English as fallback
                        self.translations[lang_code][key] = self._auto_translate(default_value, lang_code)
            
            # Save updated translations
            self.save_translations(lang_code)
    
    def _auto_translate(self, text: str, target_language: str) -> str:
        """Auto-translate text (placeholder - would integrate with translation service)"""
        
        # Placeholder translations for demo purposes
        translations = {
            'zh': {
                'Dashboard': '仪表板',
                'Analytics': '分析',
                'Models': '模型',
                'Risks': '风险',
                'Settings': '设置',
                'Help': '帮助',
                'LLM Risk Visualizer': 'LLM风险可视化工具',
                'Welcome to LLM Risk Visualizer': '欢迎使用LLM风险可视化工具',
                'Overview': '概览',
                'Recent Activity': '最近活动',
                'Statistics': '统计',
                'Risk Assessment': '风险评估',
                'Low Risk': '低风险',
                'Medium Risk': '中等风险',
                'High Risk': '高风险',
                'Critical Risk': '严重风险',
                'Save': '保存',
                'Cancel': '取消',
                'Delete': '删除',
                'Loading...': '加载中...',
                'Error': '错误',
                'Success': '成功'
            },
            'es': {
                'Dashboard': 'Panel de Control',
                'Analytics': 'Análisis',
                'Models': 'Modelos',
                'Risks': 'Riesgos',
                'Settings': 'Configuración',
                'Help': 'Ayuda',
                'LLM Risk Visualizer': 'Visualizador de Riesgos LLM',
                'Welcome to LLM Risk Visualizer': 'Bienvenido al Visualizador de Riesgos LLM',
                'Overview': 'Resumen',
                'Recent Activity': 'Actividad Reciente',
                'Statistics': 'Estadísticas',
                'Risk Assessment': 'Evaluación de Riesgos',
                'Low Risk': 'Riesgo Bajo',
                'Medium Risk': 'Riesgo Medio',
                'High Risk': 'Riesgo Alto',
                'Critical Risk': 'Riesgo Crítico',
                'Save': 'Guardar',
                'Cancel': 'Cancelar',
                'Delete': 'Eliminar',
                'Loading...': 'Cargando...',
                'Error': 'Error',
                'Success': 'Éxito'
            },
            'fr': {
                'Dashboard': 'Tableau de Bord',
                'Analytics': 'Analyses',
                'Models': 'Modèles',
                'Risks': 'Risques',
                'Settings': 'Paramètres',
                'Help': 'Aide',
                'LLM Risk Visualizer': 'Visualiseur de Risques LLM',
                'Welcome to LLM Risk Visualizer': 'Bienvenue dans le Visualiseur de Risques LLM',
                'Overview': 'Aperçu',
                'Recent Activity': 'Activité Récente',
                'Statistics': 'Statistiques',
                'Risk Assessment': 'Évaluation des Risques',
                'Low Risk': 'Risque Faible',
                'Medium Risk': 'Risque Moyen',
                'High Risk': 'Risque Élevé',
                'Critical Risk': 'Risque Critique',
                'Save': 'Enregistrer',
                'Cancel': 'Annuler',
                'Delete': 'Supprimer',
                'Loading...': 'Chargement...',
                'Error': 'Erreur',
                'Success': 'Succès'
            }
        }
        
        # Try to find translation
        if target_language in translations and text in translations[target_language]:
            return translations[target_language][text]
        
        # Fallback to original text
        return text
    
    def get_text(self, key: str, **kwargs) -> str:
        """Get translated text for a key with optional formatting"""
        
        # Try current language first
        if (self.current_language in self.translations and 
            key in self.translations[self.current_language]):
            text = self.translations[self.current_language][key]
        
        # Fallback to English
        elif (self.fallback_language in self.translations and 
              key in self.translations[self.fallback_language]):
            text = self.translations[self.fallback_language][key]
        
        # Ultimate fallback
        else:
            text = key.split('.')[-1].replace('_', ' ').title()
        
        # Apply formatting if kwargs provided
        if kwargs:
            try:
                text = text.format(**kwargs)
            except (KeyError, ValueError):
                pass  # Return unformatted text if formatting fails
        
        return text
    
    def set_language(self, language_code: str):
        """Set current language"""
        if language_code in self.supported_languages:
            self.current_language = language_code
        else:
            print(f"Unsupported language: {language_code}")
    
    def get_current_language(self) -> str:
        """Get current language code"""
        return self.current_language
    
    def get_language_config(self, language_code: str = None) -> LanguageConfig:
        """Get language configuration"""
        lang_code = language_code or self.current_language
        return self.supported_languages.get(lang_code, self.supported_languages['en'])
    
    def get_supported_languages(self) -> Dict[str, LanguageConfig]:
        """Get all supported languages"""
        return self.supported_languages.copy()
    
    def add_translation(self, language_code: str, key: str, value: str):
        """Add or update a translation"""
        if language_code not in self.translations:
            self.translations[language_code] = {}
        
        self.translations[language_code][key] = value
        self.save_translations(language_code)
    
    def get_language_completion_percentage(self, language_code: str) -> float:
        """Get translation completion percentage for a language"""
        if language_code not in self.translations:
            return 0.0
        
        english_keys = set(self.translations.get('en', {}).keys())
        target_keys = set(self.translations.get(language_code, {}).keys())
        
        if not english_keys:
            return 100.0
        
        return len(target_keys.intersection(english_keys)) / len(english_keys) * 100

class LocalizationManager:
    """Manages localization of dates, numbers, and currencies"""
    
    def __init__(self, translation_manager: TranslationManager):
        self.translation_manager = translation_manager
    
    def format_date(self, date: datetime, format_type: str = "short") -> str:
        """Format date according to current locale"""
        config = self.translation_manager.get_language_config()
        
        if format_type == "short":
            return date.strftime(config.date_format)
        elif format_type == "long":
            # This would use babel for more sophisticated formatting
            return date.strftime("%A, %B %d, %Y")
        else:
            return date.strftime(config.date_format)
    
    def format_time(self, time: datetime) -> str:
        """Format time according to current locale"""
        config = self.translation_manager.get_language_config()
        return time.strftime(config.time_format)
    
    def format_datetime(self, dt: datetime) -> str:
        """Format datetime according to current locale"""
        return f"{self.format_date(dt)} {self.format_time(dt)}"
    
    def format_number(self, number: float, decimal_places: int = 2) -> str:
        """Format number according to current locale"""
        config = self.translation_manager.get_language_config()
        
        # Format with specified decimal places
        formatted = f"{number:.{decimal_places}f}"
        
        # Split into integer and decimal parts
        parts = formatted.split('.')
        integer_part = parts[0]
        decimal_part = parts[1] if len(parts) > 1 else ""
        
        # Add thousands separators
        if len(integer_part) > 3:
            integer_with_separators = ""
            for i, digit in enumerate(reversed(integer_part)):
                if i > 0 and i % 3 == 0:
                    integer_with_separators = config.thousands_separator + integer_with_separators
                integer_with_separators = digit + integer_with_separators
            integer_part = integer_with_separators
        
        # Combine with appropriate decimal separator
        if decimal_part and int(decimal_part) > 0:
            return f"{integer_part}{config.decimal_separator}{decimal_part}"
        else:
            return integer_part
    
    def format_currency(self, amount: float, currency_code: str = None) -> str:
        """Format currency according to current locale"""
        config = self.translation_manager.get_language_config()
        currency_symbol = currency_code or config.currency_symbol
        
        formatted_number = self.format_number(amount, 2)
        
        # Different languages have different currency placement rules
        if config.code in ['en', 'zh', 'ja', 'ko']:
            return f"{currency_symbol}{formatted_number}"
        else:
            return f"{formatted_number} {currency_symbol}"
    
    def format_percentage(self, value: float, decimal_places: int = 1) -> str:
        """Format percentage according to current locale"""
        formatted_number = self.format_number(value * 100, decimal_places)
        return f"{formatted_number}%"
    
    def get_relative_time(self, dt: datetime) -> str:
        """Get relative time string (e.g., '2 hours ago')"""
        now = datetime.now()
        diff = now - dt
        
        if diff.days > 0:
            if diff.days == 1:
                return self.translation_manager.get_text('date.yesterday')
            elif diff.days < 7:
                return f"{diff.days} {self.translation_manager.get_text('time.days_ago', count=diff.days)}"
            else:
                return self.format_date(dt)
        
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} {self.translation_manager.get_text('time.hours_ago')}"
        
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} {self.translation_manager.get_text('time.minutes_ago')}"
        
        else:
            return self.translation_manager.get_text('time.seconds_ago')

class RTLSupport:
    """Provides Right-to-Left (RTL) language support"""
    
    def __init__(self, translation_manager: TranslationManager):
        self.translation_manager = translation_manager
    
    def is_rtl_language(self, language_code: str = None) -> bool:
        """Check if current or specified language is RTL"""
        config = self.translation_manager.get_language_config(language_code)
        return config.rtl
    
    def get_rtl_css(self) -> str:
        """Get CSS for RTL layout"""
        if not self.is_rtl_language():
            return ""
        
        return """
        <style>
        .rtl-content {
            direction: rtl;
            text-align: right;
        }
        
        .rtl-content .stSelectbox > div > div {
            text-align: right;
        }
        
        .rtl-content .stTextInput > div > div > input {
            text-align: right;
        }
        
        .rtl-content .stTextArea > div > div > textarea {
            text-align: right;
        }
        
        .rtl-content .metric-container {
            text-align: right;
        }
        
        .rtl-content .stDataFrame {
            direction: ltr; /* Keep data tables LTR for readability */
        }
        
        .rtl-content .stPlotlyChart {
            direction: ltr; /* Keep charts LTR */
        }
        </style>
        """
    
    def wrap_rtl_content(self, content: str) -> str:
        """Wrap content with RTL div if needed"""
        if self.is_rtl_language():
            return f'<div class="rtl-content">{content}</div>'
        return content

# Translation utilities
def _(key: str, **kwargs) -> str:
    """Shorthand function for getting translated text"""
    if 'translation_manager' in st.session_state:
        return st.session_state.translation_manager.get_text(key, **kwargs)
    return key

def t(key: str, **kwargs) -> str:
    """Alternative shorthand for translations"""
    return _(key, **kwargs)

# Streamlit Integration Functions

def initialize_i18n():
    """Initialize internationalization system"""
    if 'translation_manager' not in st.session_state:
        st.session_state.translation_manager = TranslationManager()
    
    if 'localization_manager' not in st.session_state:
        st.session_state.localization_manager = LocalizationManager(
            st.session_state.translation_manager
        )
    
    if 'rtl_support' not in st.session_state:
        st.session_state.rtl_support = RTLSupport(
            st.session_state.translation_manager
        )
    
    # Apply RTL CSS if needed
    rtl_css = st.session_state.rtl_support.get_rtl_css()
    if rtl_css:
        st.markdown(rtl_css, unsafe_allow_html=True)
    
    return (st.session_state.translation_manager, 
            st.session_state.localization_manager, 
            st.session_state.rtl_support)

def render_language_selector():
    """Render language selector widget"""
    translation_manager, _, _ = initialize_i18n()
    
    # Get supported languages
    languages = translation_manager.get_supported_languages()
    current_lang = translation_manager.get_current_language()
    
    # Create options for selectbox
    options = []
    option_mapping = {}
    
    for code, config in languages.items():
        display_name = f"{config.flag} {config.native_name}"
        options.append(display_name)
        option_mapping[display_name] = code
    
    # Current selection
    current_display = None
    for display, code in option_mapping.items():
        if code == current_lang:
            current_display = display
            break
    
    # Language selector
    selected_display = st.selectbox(
        _('settings.language'),
        options,
        index=options.index(current_display) if current_display else 0,
        key="language_selector"
    )
    
    # Update language if changed
    selected_code = option_mapping[selected_display]
    if selected_code != current_lang:
        translation_manager.set_language(selected_code)
        st.rerun()

def render_i18n_dashboard():
    """Render internationalization management dashboard"""
    st.header("🌍 " + _('settings.language'))
    
    translation_manager, localization_manager, rtl_support = initialize_i18n()
    
    # Language selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_language_selector()
    
    with col2:
        current_lang = translation_manager.get_current_language()
        config = translation_manager.get_language_config()
        st.write(f"**{_('common.current')}:** {config.flag} {config.native_name}")
    
    # Tabs for different i18n aspects
    tab1, tab2, tab3, tab4 = st.tabs([
        _('common.overview'),
        _('settings.translations'), 
        _('settings.localization'),
        _('settings.rtl_support')
    ])
    
    with tab1:
        st.subheader(_('common.overview'))
        
        # Language statistics
        languages = translation_manager.get_supported_languages()
        
        completion_data = []
        for code, config in languages.items():
            completion = translation_manager.get_language_completion_percentage(code)
            completion_data.append({
                'Language': f"{config.flag} {config.native_name}",
                'Code': code,
                'Completion': completion,
                'Current': code == translation_manager.get_current_language()
            })
        
        completion_df = pd.DataFrame(completion_data)
        
        # Completion chart
        if not completion_df.empty:
            import plotly.express as px
            
            fig = px.bar(
                completion_df,
                x='Language',
                y='Completion',
                title=_('i18n.completion_status'),
                color='Completion',
                color_continuous_scale='RdYlGn'
            )
            
            fig.update_layout(
                yaxis_title=_('i18n.completion_percentage'),
                xaxis_title=_('settings.language')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Language details table
        st.subheader(_('i18n.language_details'))
        
        details_data = []
        for code, config in languages.items():
            details_data.append({
                _('settings.language'): f"{config.flag} {config.native_name}",
                _('common.code'): code,
                _('i18n.rtl'): _('common.yes') if config.rtl else _('common.no'),
                _('i18n.date_format'): config.date_format,
                _('i18n.currency'): config.currency_symbol,
                _('i18n.completion'): f"{translation_manager.get_language_completion_percentage(code):.1f}%"
            })
        
        details_df = pd.DataFrame(details_data)
        st.dataframe(details_df, use_container_width=True)
    
    with tab2:
        st.subheader(_('settings.translations'))
        
        # Translation editor
        st.write("**" + _('i18n.translation_editor') + ":**")
        
        # Language selection for editing
        edit_languages = list(translation_manager.get_supported_languages().keys())
        edit_lang = st.selectbox(
            _('i18n.select_language_to_edit'),
            edit_languages,
            format_func=lambda x: f"{translation_manager.get_language_config(x).flag} {translation_manager.get_language_config(x).native_name}"
        )
        
        # Get translations for selected language
        translations = translation_manager.translations.get(edit_lang, {})
        
        if translations:
            # Search functionality
            search_term = st.text_input(_('common.search') + " " + _('settings.translations'))
            
            # Filter translations based on search
            if search_term:
                filtered_translations = {
                    k: v for k, v in translations.items() 
                    if search_term.lower() in k.lower() or search_term.lower() in v.lower()
                }
            else:
                filtered_translations = translations
            
            # Display translations in editable format
            if filtered_translations:
                st.write(f"**{len(filtered_translations)}** " + _('i18n.translations_found'))
                
                # Show first 20 translations for editing
                for i, (key, value) in enumerate(list(filtered_translations.items())[:20]):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.text(key)
                    
                    with col2:
                        new_value = st.text_input(
                            "",
                            value=value,
                            key=f"trans_{edit_lang}_{key}_{i}",
                            label_visibility="collapsed"
                        )
                        
                        if new_value != value:
                            translation_manager.add_translation(edit_lang, key, new_value)
                
                if len(filtered_translations) > 20:
                    st.info(f"{_('common.showing')} 20 {_('common.of')} {len(filtered_translations)} {_('settings.translations')}")
            else:
                st.info(_('message.no_data'))
        else:
            st.warning(_('i18n.no_translations_found'))
        
        # Add new translation
        with st.expander(_('i18n.add_new_translation')):
            new_key = st.text_input(_('i18n.translation_key'))
            new_value = st.text_input(_('i18n.translation_value'))
            
            if st.button(_('common.add')) and new_key and new_value:
                translation_manager.add_translation(edit_lang, new_key, new_value)
                st.success(_('i18n.translation_added'))
                st.rerun()
    
    with tab3:
        st.subheader(_('settings.localization'))
        
        # Localization examples
        current_config = translation_manager.get_language_config()
        
        st.write("**" + _('i18n.formatting_examples') + ":**")
        
        # Date formatting
        sample_date = datetime.now()
        formatted_date = localization_manager.format_date(sample_date)
        st.write(f"**{_('common.date')}:** {formatted_date}")
        
        # Time formatting
        formatted_time = localization_manager.format_time(sample_date)
        st.write(f"**{_('common.time')}:** {formatted_time}")
        
        # Number formatting
        sample_number = 1234567.89
        formatted_number = localization_manager.format_number(sample_number)
        st.write(f"**{_('common.number')}:** {formatted_number}")
        
        # Currency formatting
        sample_amount = 1234.56
        formatted_currency = localization_manager.format_currency(sample_amount)
        st.write(f"**{_('common.currency')}:** {formatted_currency}")
        
        # Percentage formatting
        sample_percentage = 0.1234
        formatted_percentage = localization_manager.format_percentage(sample_percentage)
        st.write(f"**{_('common.percentage')}:** {formatted_percentage}")
        
        # Relative time
        past_time = datetime.now() - timedelta(hours=2, minutes=30)
        relative_time = localization_manager.get_relative_time(past_time)
        st.write(f"**{_('common.relative_time')}:** {relative_time}")
        
        # Localization settings
        st.write("**" + _('i18n.localization_settings') + ":**")
        
        with st.form("localization_settings"):
            new_date_format = st.text_input(_('i18n.date_format'), value=current_config.date_format)
            new_time_format = st.text_input(_('i18n.time_format'), value=current_config.time_format)
            new_currency_symbol = st.text_input(_('i18n.currency_symbol'), value=current_config.currency_symbol)
            new_decimal_separator = st.text_input(_('i18n.decimal_separator'), value=current_config.decimal_separator)
            new_thousands_separator = st.text_input(_('i18n.thousands_separator'), value=current_config.thousands_separator)
            
            if st.form_submit_button(_('common.save')):
                # Update configuration (in a real app, this would be persisted)
                current_config.date_format = new_date_format
                current_config.time_format = new_time_format
                current_config.currency_symbol = new_currency_symbol
                current_config.decimal_separator = new_decimal_separator
                current_config.thousands_separator = new_thousands_separator
                
                st.success(_('i18n.settings_updated'))
                st.rerun()
    
    with tab4:
        st.subheader(_('i18n.rtl_support'))
        
        # RTL status
        is_rtl = rtl_support.is_rtl_language()
        st.write(f"**{_('i18n.current_direction')}:** {_('i18n.rtl') if is_rtl else _('i18n.ltr')}")
        
        if is_rtl:
            st.info(_('i18n.rtl_enabled_notice'))
            
            # RTL demo content
            st.write("**" + _('i18n.rtl_demo') + ":**")
            
            rtl_demo_html = rtl_support.wrap_rtl_content(
                f"""
                <div style="border: 1px solid #ccc; padding: 15px; margin: 10px 0;">
                    <h3>{_('dashboard.title')}</h3>
                    <p>{_('dashboard.welcome')}</p>
                    <ul>
                        <li>{_('nav.dashboard')}</li>
                        <li>{_('nav.analytics')}</li>
                        <li>{_('nav.models')}</li>
                        <li>{_('nav.risks')}</li>
                    </ul>
                </div>
                """
            )
            
            st.markdown(rtl_demo_html, unsafe_allow_html=True)
        else:
            st.info(_('i18n.ltr_language_notice'))
        
        # RTL language list
        st.write("**" + _('i18n.rtl_languages') + ":**")
        
        rtl_languages = [
            (code, config) for code, config in translation_manager.get_supported_languages().items()
            if config.rtl
        ]
        
        if rtl_languages:
            for code, config in rtl_languages:
                st.write(f"• {config.flag} {config.native_name} ({code})")
        else:
            st.write(_('i18n.no_rtl_languages'))

def create_localized_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create localized version of dataframe"""
    if 'localization_manager' not in st.session_state:
        return df
    
    localization_manager = st.session_state.localization_manager
    localized_df = df.copy()
    
    # Localize numeric columns
    for col in localized_df.columns:
        if localized_df[col].dtype in ['float64', 'int64']:
            if 'rate' in col.lower() or 'percentage' in col.lower():
                # Format as percentage
                localized_df[col] = localized_df[col].apply(
                    lambda x: localization_manager.format_percentage(x / 100) if pd.notna(x) else ""
                )
            elif 'amount' in col.lower() or 'cost' in col.lower() or 'price' in col.lower():
                # Format as currency
                localized_df[col] = localized_df[col].apply(
                    lambda x: localization_manager.format_currency(x) if pd.notna(x) else ""
                )
            else:
                # Format as number
                localized_df[col] = localized_df[col].apply(
                    lambda x: localization_manager.format_number(x) if pd.notna(x) else ""
                )
        
        elif 'date' in col.lower() and pd.api.types.is_datetime64_any_dtype(localized_df[col]):
            # Format dates
            localized_df[col] = localized_df[col].apply(
                lambda x: localization_manager.format_date(x) if pd.notna(x) else ""
            )
    
    return localized_df

if __name__ == "__main__":
    # Example usage and testing
    
    # Initialize translation manager
    translation_manager = TranslationManager()
    localization_manager = LocalizationManager(translation_manager)
    rtl_support = RTLSupport(translation_manager)
    
    # Test translations
    print("Testing translations...")
    
    # Test English
    translation_manager.set_language('en')
    print(f"English: {translation_manager.get_text('dashboard.title')}")
    
    # Test Chinese
    translation_manager.set_language('zh')
    print(f"Chinese: {translation_manager.get_text('dashboard.title')}")
    
    # Test Spanish
    translation_manager.set_language('es')
    print(f"Spanish: {translation_manager.get_text('dashboard.title')}")
    
    # Test localization
    print("\nTesting localization...")
    
    sample_date = datetime.now()
    sample_number = 1234567.89
    sample_currency = 1234.56
    
    for lang in ['en', 'zh', 'es', 'fr', 'de']:
        translation_manager.set_language(lang)
        config = translation_manager.get_language_config()
        
        print(f"\n{config.flag} {config.native_name} ({lang}):")
        print(f"  Date: {localization_manager.format_date(sample_date)}")
        print(f"  Number: {localization_manager.format_number(sample_number)}")
        print(f"  Currency: {localization_manager.format_currency(sample_currency)}")
    
    # Test RTL support
    print("\nTesting RTL support...")
    
    translation_manager.set_language('ar')
    print(f"Arabic RTL: {rtl_support.is_rtl_language()}")
    
    translation_manager.set_language('en')
    print(f"English RTL: {rtl_support.is_rtl_language()}")
    
    # Test completion percentages
    print("\nTranslation completion:")
    for lang_code in translation_manager.get_supported_languages().keys():
        completion = translation_manager.get_language_completion_percentage(lang_code)
        config = translation_manager.get_language_config(lang_code)
        print(f"  {config.flag} {config.native_name}: {completion:.1f}%")
    
    print("\nInternationalization module test completed!")