"""Internationalization and localization utilities."""

import json
import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import re

logger = logging.getLogger(__name__)


@dataclass
class LocaleConfig:
    """Configuration for a specific locale."""
    language: str
    country: str
    currency: str
    timezone: str
    date_format: str
    number_format: str
    decimal_separator: str
    thousands_separator: str


class TranslationManager:
    """Manage translations for different languages."""
    
    def __init__(self, default_locale: str = "en_US"):
        self.default_locale = default_locale
        self.current_locale = default_locale
        self.translations: Dict[str, Dict[str, str]] = {}
        
        # Initialize with basic translations
        self._initialize_translations()
        
    def _initialize_translations(self):
        """Initialize with basic power system terminology."""
        
        # English (US)
        self.translations["en_US"] = {
            # General terms
            "voltage": "Voltage",
            "current": "Current", 
            "power": "Power",
            "frequency": "Frequency",
            "energy": "Energy",
            "load": "Load",
            "generation": "Generation",
            "transformer": "Transformer",
            "line": "Line",
            "bus": "Bus",
            
            # Units
            "voltage_unit": "V",
            "current_unit": "A",
            "power_unit_kw": "kW",
            "power_unit_mw": "MW",
            "frequency_unit": "Hz",
            "energy_unit": "kWh",
            
            # Status messages
            "simulation_started": "Simulation started",
            "simulation_completed": "Simulation completed",
            "power_flow_converged": "Power flow converged",
            "power_flow_failed": "Power flow failed to converge",
            "constraint_violation": "Constraint violation detected",
            "emergency_action": "Emergency action required",
            
            # Errors
            "invalid_input": "Invalid input provided",
            "network_error": "Network topology error",
            "computation_error": "Computation error occurred",
            "file_not_found": "File not found",
            "permission_denied": "Permission denied"
        }
        
        # Spanish
        self.translations["es_ES"] = {
            "voltage": "Voltaje",
            "current": "Corriente",
            "power": "Potencia", 
            "frequency": "Frecuencia",
            "energy": "Energía",
            "load": "Carga",
            "generation": "Generación",
            "transformer": "Transformador",
            "line": "Línea",
            "bus": "Barra",
            
            "voltage_unit": "V",
            "current_unit": "A", 
            "power_unit_kw": "kW",
            "power_unit_mw": "MW",
            "frequency_unit": "Hz",
            "energy_unit": "kWh",
            
            "simulation_started": "Simulación iniciada",
            "simulation_completed": "Simulación completada",
            "power_flow_converged": "Flujo de potencia convergió",
            "power_flow_failed": "Flujo de potencia falló en converger",
            "constraint_violation": "Violación de restricción detectada",
            "emergency_action": "Acción de emergencia requerida",
            
            "invalid_input": "Entrada inválida proporcionada",
            "network_error": "Error de topología de red",
            "computation_error": "Error de cálculo ocurrido",
            "file_not_found": "Archivo no encontrado",
            "permission_denied": "Permiso denegado"
        }
        
        # French
        self.translations["fr_FR"] = {
            "voltage": "Tension",
            "current": "Courant",
            "power": "Puissance",
            "frequency": "Fréquence",
            "energy": "Énergie", 
            "load": "Charge",
            "generation": "Génération",
            "transformer": "Transformateur",
            "line": "Ligne",
            "bus": "Jeu de barres",
            
            "voltage_unit": "V",
            "current_unit": "A",
            "power_unit_kw": "kW",
            "power_unit_mw": "MW", 
            "frequency_unit": "Hz",
            "energy_unit": "kWh",
            
            "simulation_started": "Simulation démarrée",
            "simulation_completed": "Simulation terminée",
            "power_flow_converged": "Répartition de charge convergée",
            "power_flow_failed": "Échec de convergence de la répartition de charge",
            "constraint_violation": "Violation de contrainte détectée",
            "emergency_action": "Action d'urgence requise"
        }
        
        # German
        self.translations["de_DE"] = {
            "voltage": "Spannung",
            "current": "Strom", 
            "power": "Leistung",
            "frequency": "Frequenz",
            "energy": "Energie",
            "load": "Last",
            "generation": "Erzeugung",
            "transformer": "Transformator",
            "line": "Leitung",
            "bus": "Sammelschiene",
            
            "voltage_unit": "V",
            "current_unit": "A",
            "power_unit_kw": "kW", 
            "power_unit_mw": "MW",
            "frequency_unit": "Hz",
            "energy_unit": "kWh",
            
            "simulation_started": "Simulation gestartet",
            "simulation_completed": "Simulation abgeschlossen", 
            "power_flow_converged": "Lastfluss konvergiert",
            "power_flow_failed": "Lastfluss konvergierte nicht",
            "constraint_violation": "Beschränkungsverletzung erkannt",
            "emergency_action": "Notfallmaßnahme erforderlich"
        }
        
        # Japanese
        self.translations["ja_JP"] = {
            "voltage": "電圧",
            "current": "電流",
            "power": "電力", 
            "frequency": "周波数",
            "energy": "エネルギー",
            "load": "負荷",
            "generation": "発電",
            "transformer": "変圧器",
            "line": "線路",
            "bus": "母線",
            
            "voltage_unit": "V",
            "current_unit": "A",
            "power_unit_kw": "kW",
            "power_unit_mw": "MW",
            "frequency_unit": "Hz", 
            "energy_unit": "kWh",
            
            "simulation_started": "シミュレーション開始",
            "simulation_completed": "シミュレーション完了",
            "power_flow_converged": "潮流計算が収束しました",
            "power_flow_failed": "潮流計算が収束しませんでした",
            "constraint_violation": "制約違反が検出されました",
            "emergency_action": "緊急対応が必要です"
        }
        
        # Simplified Chinese
        self.translations["zh_CN"] = {
            "voltage": "电压", 
            "current": "电流",
            "power": "功率",
            "frequency": "频率",
            "energy": "能量",
            "load": "负荷",
            "generation": "发电",
            "transformer": "变压器",
            "line": "线路",
            "bus": "母线",
            
            "voltage_unit": "V",
            "current_unit": "A",
            "power_unit_kw": "kW",
            "power_unit_mw": "MW",
            "frequency_unit": "Hz",
            "energy_unit": "kWh",
            
            "simulation_started": "仿真开始",
            "simulation_completed": "仿真完成",
            "power_flow_converged": "潮流计算收敛",
            "power_flow_failed": "潮流计算未收敛", 
            "constraint_violation": "检测到约束违规",
            "emergency_action": "需要紧急行动"
        }
        
        # French
        self.translations["fr_FR"] = {
            "voltage": "tension",
            "current": "Courant",
            "power": "Puissance",
            "frequency": "Fréquence", 
            "energy": "Énergie",
            "load": "Charge",
            "generation": "Génération",
            "transformer": "Transformateur",
            "line": "Ligne",
            "bus": "Jeu de barres",
            
            "voltage_unit": "V",
            "current_unit": "A",
            "power_unit_kw": "kW", 
            "power_unit_mw": "MW",
            "frequency_unit": "Hz",
            "energy_unit": "kWh",
            
            "simulation_started": "Simulation démarrée",
            "simulation_completed": "Simulation terminée",
            "power_flow_converged": "Écoulement de puissance convergé",
            "power_flow_failed": "L'écoulement de puissance n'a pas convergé",
            "constraint_violation": "Violation de contrainte détectée",
            "emergency_action": "Action d'urgence requise",
            
            "invalid_input": "Entrée invalide fournie",
            "network_error": "Erreur de topologie de réseau",
            "computation_error": "Erreur de calcul survenue", 
            "file_not_found": "Fichier non trouvé",
            "permission_denied": "Permission refusée"
        }
        
        # German
        self.translations["de_DE"] = {
            "voltage": "Spannung",
            "current": "Strom",
            "power": "Leistung",
            "frequency": "Frequenz",
            "energy": "Energie", 
            "load": "Last",
            "generation": "Erzeugung",
            "transformer": "Transformator",
            "line": "Leitung",
            "bus": "Sammelschiene",
            
            "voltage_unit": "V",
            "current_unit": "A",
            "power_unit_kw": "kW",
            "power_unit_mw": "MW", 
            "frequency_unit": "Hz",
            "energy_unit": "kWh",
            
            "simulation_started": "Simulation gestartet",
            "simulation_completed": "Simulation abgeschlossen",
            "power_flow_converged": "Lastfluss konvergiert",
            "power_flow_failed": "Lastfluss nicht konvergiert",
            "constraint_violation": "Nebenbedingungsverletzung erkannt",
            "emergency_action": "Notfallaktion erforderlich",
            
            "invalid_input": "Ungültige Eingabe bereitgestellt",
            "network_error": "Netzwerktopologie-Fehler",
            "computation_error": "Berechnungsfehler aufgetreten",
            "file_not_found": "Datei nicht gefunden",
            "permission_denied": "Zugriff verweigert"
        }
        
        # Japanese
        self.translations["ja_JP"] = {
            "voltage": "電圧",
            "current": "電流", 
            "power": "電力",
            "frequency": "周波数",
            "energy": "エネルギー",
            "load": "負荷",
            "generation": "発電",
            "transformer": "変圧器",
            "line": "線路",
            "bus": "母線",
            
            "voltage_unit": "V",
            "current_unit": "A",
            "power_unit_kw": "kW",
            "power_unit_mw": "MW",
            "frequency_unit": "Hz", 
            "energy_unit": "kWh",
            
            "simulation_started": "シミュレーション開始",
            "simulation_completed": "シミュレーション完了",
            "power_flow_converged": "潮流計算収束",
            "power_flow_failed": "潮流計算が収束しませんでした",
            "constraint_violation": "制約違反が検出されました",
            "emergency_action": "緊急対応が必要です",
            
            "invalid_input": "無効な入力が提供されました",
            "network_error": "ネットワークトポロジーエラー",
            "computation_error": "計算エラーが発生しました",
            "file_not_found": "ファイルが見つかりません",
            "permission_denied": "アクセス権限がありません"
        }
        
        # Chinese (Simplified)
        self.translations["zh_CN"] = {
            "voltage": "电压",
            "current": "电流",
            "power": "功率", 
            "frequency": "频率",
            "energy": "电能",
            "load": "负荷",
            "generation": "发电",
            "transformer": "变压器",
            "line": "线路",
            "bus": "母线",
            
            "voltage_unit": "V",
            "current_unit": "A",
            "power_unit_kw": "kW",
            "power_unit_mw": "MW",
            "frequency_unit": "Hz",
            "energy_unit": "kWh",
            
            "simulation_started": "仿真已开始",
            "simulation_completed": "仿真已完成",
            "power_flow_converged": "潮流计算收敛",
            "power_flow_failed": "潮流计算未收敛",
            "constraint_violation": "检测到约束违反",
            "emergency_action": "需要紧急措施",
            
            "invalid_input": "提供的输入无效",
            "network_error": "网络拓扑错误",
            "computation_error": "发生计算错误",
            "file_not_found": "未找到文件",
            "permission_denied": "权限被拒绝"
        }
    
    def set_locale(self, locale: str) -> bool:
        """Set the current locale."""
        if locale in self.translations:
            self.current_locale = locale
            logger.info(f"Locale set to: {locale}")
            return True
        else:
            logger.warning(f"Locale not supported: {locale}")
            return False
    
    def translate(self, key: str, locale: Optional[str] = None) -> str:
        """Get translation for a key."""
        target_locale = locale or self.current_locale
        
        # Try requested locale
        if target_locale in self.translations:
            if key in self.translations[target_locale]:
                return self.translations[target_locale][key]
        
        # Fallback to default locale
        if self.default_locale in self.translations:
            if key in self.translations[self.default_locale]:
                return self.translations[self.default_locale][key]
        
        # Return key if no translation found
        logger.warning(f"No translation found for key: {key}")
        return key
    
    def get_available_locales(self) -> List[str]:
        """Get list of available locales."""
        return list(self.translations.keys())
    
    def add_translations(self, locale: str, translations: Dict[str, str]) -> None:
        """Add translations for a locale."""
        if locale not in self.translations:
            self.translations[locale] = {}
        
        self.translations[locale].update(translations)
        logger.info(f"Added {len(translations)} translations for {locale}")


class LocaleManager:
    """Manage locale-specific formatting and configurations."""
    
    def __init__(self):
        self.locale_configs = {
            "en_US": LocaleConfig(
                language="en", country="US", currency="USD", timezone="UTC",
                date_format="%m/%d/%Y", number_format="1,234.56",
                decimal_separator=".", thousands_separator=","
            ),
            "es_ES": LocaleConfig(
                language="es", country="ES", currency="EUR", timezone="Europe/Madrid",
                date_format="%d/%m/%Y", number_format="1.234,56",
                decimal_separator=",", thousands_separator="."
            ),
            "fr_FR": LocaleConfig(
                language="fr", country="FR", currency="EUR", timezone="Europe/Paris", 
                date_format="%d/%m/%Y", number_format="1 234,56",
                decimal_separator=",", thousands_separator=" "
            ),
            "de_DE": LocaleConfig(
                language="de", country="DE", currency="EUR", timezone="Europe/Berlin",
                date_format="%d.%m.%Y", number_format="1.234,56",
                decimal_separator=",", thousands_separator="."
            ),
            "ja_JP": LocaleConfig(
                language="ja", country="JP", currency="JPY", timezone="Asia/Tokyo",
                date_format="%Y/%m/%d", number_format="1,234.56",
                decimal_separator=".", thousands_separator=","
            ),
            "zh_CN": LocaleConfig(
                language="zh", country="CN", currency="CNY", timezone="Asia/Shanghai",
                date_format="%Y/%m/%d", number_format="1,234.56",
                decimal_separator=".", thousands_separator=","
            )
        }
        
        self.current_locale = "en_US"
    
    def set_locale(self, locale: str) -> bool:
        """Set current locale."""
        if locale in self.locale_configs:
            self.current_locale = locale
            return True
        return False
    
    def format_number(self, number: float, locale: Optional[str] = None) -> str:
        """Format number according to locale."""
        target_locale = locale or self.current_locale
        config = self.locale_configs.get(target_locale, self.locale_configs["en_US"])
        
        # Basic number formatting
        if abs(number) >= 1000000:
            # Millions
            formatted = f"{number/1000000:.2f}M"
        elif abs(number) >= 1000:
            # Thousands  
            formatted = f"{number/1000:.1f}k"
        else:
            formatted = f"{number:.2f}"
        
        # Apply locale-specific separators
        if config.decimal_separator != ".":
            formatted = formatted.replace(".", config.decimal_separator)
        
        return formatted
    
    def format_currency(self, amount: float, locale: Optional[str] = None) -> str:
        """Format currency according to locale."""
        target_locale = locale or self.current_locale
        config = self.locale_configs.get(target_locale, self.locale_configs["en_US"])
        
        formatted_number = self.format_number(amount, target_locale)
        
        currency_symbols = {
            "USD": "$", "EUR": "€", "JPY": "¥", "CNY": "¥"
        }
        
        symbol = currency_symbols.get(config.currency, config.currency)
        
        if config.currency in ["USD"]:
            return f"{symbol}{formatted_number}"
        else:
            return f"{formatted_number} {symbol}"
    
    def format_date(self, date: datetime, locale: Optional[str] = None) -> str:
        """Format date according to locale."""
        target_locale = locale or self.current_locale
        config = self.locale_configs.get(target_locale, self.locale_configs["en_US"])
        
        return date.strftime(config.date_format)
    
    def get_timezone(self, locale: Optional[str] = None) -> str:
        """Get timezone for locale."""
        target_locale = locale or self.current_locale
        config = self.locale_configs.get(target_locale, self.locale_configs["en_US"])
        return config.timezone


class MultiRegionSupport:
    """Support for multi-region deployment and compliance."""
    
    def __init__(self):
        self.regional_configs = {
            "north_america": {
                "frequency_standard": 60.0,  # Hz
                "voltage_standards": [120, 240, 480, 4160, 13800, 69000],  # V
                "safety_standards": ["IEEE", "NERC", "NEC"],
                "regulatory_requirements": ["FERC", "NERC_CIP"],
                "privacy_regulations": ["CPPA"],
                "default_locale": "en_US"
            },
            "europe": {
                "frequency_standard": 50.0,  # Hz  
                "voltage_standards": [230, 400, 690, 6600, 11000, 20000],  # V
                "safety_standards": ["IEC", "EN", "CENELEC"],
                "regulatory_requirements": ["EU_GDPR", "EU_NIS"],
                "privacy_regulations": ["GDPR"],
                "default_locale": "en_GB"
            },
            "asia_pacific": {
                "frequency_standard": 50.0,  # Hz (mostly, Japan is mixed)
                "voltage_standards": [220, 380, 660, 6600, 11000, 22000],  # V
                "safety_standards": ["IEC", "JIS", "GB"],
                "regulatory_requirements": ["PDPA", "PIPL"],
                "privacy_regulations": ["PDPA"],
                "default_locale": "en_US"  # English as lingua franca
            },
            "japan": {
                "frequency_standard": [50.0, 60.0],  # Both frequencies used
                "voltage_standards": [100, 200, 400, 6600, 22000],  # V
                "safety_standards": ["JIS", "JEC"],
                "regulatory_requirements": ["METI", "ESCJ"],
                "privacy_regulations": ["APPI"],
                "default_locale": "ja_JP"
            },
            "china": {
                "frequency_standard": 50.0,  # Hz
                "voltage_standards": [220, 380, 660, 6000, 10000, 35000],  # V
                "safety_standards": ["GB", "IEC"],
                "regulatory_requirements": ["PIPL", "CSL"],
                "privacy_regulations": ["PIPL"],
                "default_locale": "zh_CN"
            }
        }
        
        self.current_region = "north_america"
    
    def set_region(self, region: str) -> bool:
        """Set current region."""
        if region in self.regional_configs:
            self.current_region = region
            logger.info(f"Region set to: {region}")
            return True
        else:
            logger.warning(f"Region not supported: {region}")
            return False
    
    def get_frequency_standard(self, region: Optional[str] = None) -> float:
        """Get frequency standard for region."""
        target_region = region or self.current_region
        config = self.regional_configs.get(target_region, self.regional_configs["north_america"])
        
        freq = config["frequency_standard"]
        if isinstance(freq, list):
            return freq[0]  # Return first frequency if multiple
        return freq
    
    def get_voltage_standards(self, region: Optional[str] = None) -> List[float]:
        """Get voltage standards for region."""
        target_region = region or self.current_region
        config = self.regional_configs.get(target_region, self.regional_configs["north_america"])
        return config["voltage_standards"]
    
    def check_compliance(self, region: Optional[str] = None) -> Dict[str, List[str]]:
        """Get compliance requirements for region."""
        target_region = region or self.current_region
        config = self.regional_configs.get(target_region, self.regional_configs["north_america"])
        
        return {
            "safety_standards": config["safety_standards"],
            "regulatory_requirements": config["regulatory_requirements"],
            "privacy_regulations": config["privacy_regulations"]
        }
    
    def get_default_locale(self, region: Optional[str] = None) -> str:
        """Get default locale for region.""" 
        target_region = region or self.current_region
        config = self.regional_configs.get(target_region, self.regional_configs["north_america"])
        return config["default_locale"]


# Global instances
global_translation_manager = TranslationManager()
global_locale_manager = LocaleManager() 
global_region_support = MultiRegionSupport()