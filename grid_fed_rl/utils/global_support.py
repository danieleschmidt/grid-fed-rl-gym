"""Global-first implementation with internationalization and multi-region support."""

import os
import json
import locale
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    KOREAN = "ko"
    ITALIAN = "it"


class Region(Enum):
    """Supported regions for deployment."""
    NORTH_AMERICA = "na"
    EUROPE = "eu"
    ASIA_PACIFIC = "apac"
    SOUTH_AMERICA = "sa"
    MIDDLE_EAST_AFRICA = "mea"
    CHINA = "cn"
    AUSTRALIA = "au"


@dataclass
class LocalizationConfig:
    """Configuration for localization settings."""
    language: SupportedLanguage = SupportedLanguage.ENGLISH
    region: Region = Region.NORTH_AMERICA
    currency: str = "USD"
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    decimal_separator: str = "."
    thousands_separator: str = ","
    timezone: str = "UTC"


class TranslationManager:
    """Manage translations and localized content."""
    
    def __init__(self):
        self.translations: Dict[str, Dict[str, str]] = {}
        self.current_language = SupportedLanguage.ENGLISH
        self._load_default_translations()
        
    def _load_default_translations(self) -> None:
        """Load default translations for key messages."""
        self.translations = {
            "en": {
                "grid_environment_created": "Grid environment created successfully",
                "power_flow_converged": "Power flow analysis converged",
                "power_flow_failed": "Power flow analysis failed to converge",
                "constraint_violation": "Constraint violation detected",
                "safety_intervention": "Safety intervention activated",
                "episode_completed": "Episode completed",
                "action_invalid": "Invalid action provided",
                "system_ready": "System ready for operation",
                "error_occurred": "An error occurred",
                "performance_warning": "Performance warning",
                "voltage_violation": "Voltage constraint violation",
                "frequency_violation": "Frequency constraint violation",
                "thermal_violation": "Thermal limit violation",
                "cache_hit": "Cache hit",
                "cache_miss": "Cache miss",
                "worker_added": "Worker added to pool",
                "worker_removed": "Worker removed from pool",
                "memory_cleanup": "Memory cleanup performed",
                "health_check_passed": "Health check passed",
                "health_check_failed": "Health check failed"
            },
            "es": {
                "grid_environment_created": "Entorno de red creado exitosamente",
                "power_flow_converged": "El análisis de flujo de potencia convergió",
                "power_flow_failed": "El análisis de flujo de potencia falló en converger",
                "constraint_violation": "Violación de restricción detectada",
                "safety_intervention": "Intervención de seguridad activada",
                "episode_completed": "Episodio completado",
                "action_invalid": "Acción inválida proporcionada",
                "system_ready": "Sistema listo para operación",
                "error_occurred": "Ocurrió un error",
                "performance_warning": "Advertencia de rendimiento",
                "voltage_violation": "Violación de restricción de voltaje",
                "frequency_violation": "Violación de restricción de frecuencia",
                "thermal_violation": "Violación de límite térmico",
                "cache_hit": "Acierto de caché",
                "cache_miss": "Fallo de caché",
                "worker_added": "Trabajador agregado al pool",
                "worker_removed": "Trabajador removido del pool",
                "memory_cleanup": "Limpieza de memoria realizada",
                "health_check_passed": "Verificación de salud exitosa",
                "health_check_failed": "Verificación de salud falló"
            },
            "fr": {
                "grid_environment_created": "Environnement de réseau créé avec succès",
                "power_flow_converged": "L'analyse de flux de puissance a convergé",
                "power_flow_failed": "L'analyse de flux de puissance a échoué à converger",
                "constraint_violation": "Violation de contrainte détectée",
                "safety_intervention": "Intervention de sécurité activée",
                "episode_completed": "Épisode terminé",
                "action_invalid": "Action invalide fournie",
                "system_ready": "Système prêt pour l'opération",
                "error_occurred": "Une erreur s'est produite",
                "performance_warning": "Avertissement de performance",
                "voltage_violation": "Violation de contrainte de tension",
                "frequency_violation": "Violation de contrainte de fréquence",
                "thermal_violation": "Violation de limite thermique",
                "cache_hit": "Succès de cache",
                "cache_miss": "Échec de cache",
                "worker_added": "Travailleur ajouté au pool",
                "worker_removed": "Travailleur retiré du pool",
                "memory_cleanup": "Nettoyage de mémoire effectué",
                "health_check_passed": "Vérification de santé réussie",
                "health_check_failed": "Vérification de santé échouée"
            },
            "de": {
                "grid_environment_created": "Netzumgebung erfolgreich erstellt",
                "power_flow_converged": "Leistungsflussanalyse konvergiert",
                "power_flow_failed": "Leistungsflussanalyse konvergierte nicht",
                "constraint_violation": "Beschränkungsverletzung erkannt",
                "safety_intervention": "Sicherheitsintervention aktiviert",
                "episode_completed": "Episode abgeschlossen",
                "action_invalid": "Ungültige Aktion bereitgestellt",
                "system_ready": "System bereit für Betrieb",
                "error_occurred": "Ein Fehler ist aufgetreten",
                "performance_warning": "Leistungswarnung",
                "voltage_violation": "Spannungsbeschränkungsverletzung",
                "frequency_violation": "Frequenzbeschränkungsverletzung",
                "thermal_violation": "Thermische Grenzverletzung",
                "cache_hit": "Cache-Treffer",
                "cache_miss": "Cache-Verfehlung",
                "worker_added": "Arbeiter zum Pool hinzugefügt",
                "worker_removed": "Arbeiter aus Pool entfernt",
                "memory_cleanup": "Speicherbereinigung durchgeführt",
                "health_check_passed": "Gesundheitsprüfung bestanden",
                "health_check_failed": "Gesundheitsprüfung fehlgeschlagen"
            },
            "ja": {
                "grid_environment_created": "グリッド環境が正常に作成されました",
                "power_flow_converged": "潮流解析が収束しました",
                "power_flow_failed": "潮流解析の収束に失敗しました",
                "constraint_violation": "制約違反が検出されました",
                "safety_intervention": "安全介入が有効化されました",
                "episode_completed": "エピソードが完了しました",
                "action_invalid": "無効なアクションが提供されました",
                "system_ready": "システムは動作準備完了です",
                "error_occurred": "エラーが発生しました",
                "performance_warning": "パフォーマンス警告",
                "voltage_violation": "電圧制約違反",
                "frequency_violation": "周波数制約違反",
                "thermal_violation": "熱制限違反",
                "cache_hit": "キャッシュヒット",
                "cache_miss": "キャッシュミス",
                "worker_added": "ワーカーがプールに追加されました",
                "worker_removed": "ワーカーがプールから削除されました",
                "memory_cleanup": "メモリクリーンアップが実行されました",
                "health_check_passed": "ヘルスチェックが成功しました",
                "health_check_failed": "ヘルスチェックが失敗しました"
            },
            "zh": {
                "grid_environment_created": "电网环境创建成功",
                "power_flow_converged": "潮流分析收敛",
                "power_flow_failed": "潮流分析收敛失败",
                "constraint_violation": "检测到约束违反",
                "safety_intervention": "安全干预已激活",
                "episode_completed": "回合完成",
                "action_invalid": "提供的动作无效",
                "system_ready": "系统准备就绪",
                "error_occurred": "发生错误",
                "performance_warning": "性能警告",
                "voltage_violation": "电压约束违反",
                "frequency_violation": "频率约束违反",
                "thermal_violation": "热限制违反",
                "cache_hit": "缓存命中",
                "cache_miss": "缓存未命中",
                "worker_added": "工作进程已添加到池中",
                "worker_removed": "工作进程已从池中移除",
                "memory_cleanup": "已执行内存清理",
                "health_check_passed": "健康检查通过",
                "health_check_failed": "健康检查失败"
            }
        }
        
    def set_language(self, language: SupportedLanguage) -> None:
        """Set current language for translations."""
        self.current_language = language
        logger.info(f"Language set to {language.value}")
        
    def get_translation(self, key: str, **kwargs) -> str:
        """Get translated text for a key."""
        lang_code = self.current_language.value
        
        if lang_code in self.translations and key in self.translations[lang_code]:
            text = self.translations[lang_code][key]
        elif key in self.translations.get("en", {}):
            # Fallback to English
            text = self.translations["en"][key]
        else:
            # Fallback to key itself
            text = key.replace("_", " ").title()
            
        # Format with any provided arguments
        try:
            return text.format(**kwargs)
        except (KeyError, ValueError):
            return text
            
    def add_translations(self, language: SupportedLanguage, translations: Dict[str, str]) -> None:
        """Add custom translations for a language."""
        lang_code = language.value
        if lang_code not in self.translations:
            self.translations[lang_code] = {}
        self.translations[lang_code].update(translations)
        
    def get_available_languages(self) -> List[str]:
        """Get list of available languages."""
        return list(self.translations.keys())


class RegionalConfig:
    """Manage regional configuration and compliance."""
    
    def __init__(self, region: Region = Region.NORTH_AMERICA):
        self.region = region
        self.config = self._get_regional_config(region)
        
    def _get_regional_config(self, region: Region) -> Dict[str, Any]:
        """Get configuration for specific region."""
        configs = {
            Region.NORTH_AMERICA: {
                "data_retention_days": 2555,  # 7 years
                "encryption_required": True,
                "audit_logging": True,
                "data_sovereignty": False,
                "privacy_framework": "CCPA",
                "currency": "USD",
                "voltage_standard": "ANSI",
                "frequency_standard": 60.0,
                "regulatory_bodies": ["NERC", "FERC"],
                "grid_codes": ["IEEE 1547", "UL 1741"],
                "time_zones": ["America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"]
            },
            Region.EUROPE: {
                "data_retention_days": 2555,  # 7 years
                "encryption_required": True,
                "audit_logging": True,
                "data_sovereignty": True,
                "privacy_framework": "GDPR",
                "currency": "EUR",
                "voltage_standard": "IEC",
                "frequency_standard": 50.0,
                "regulatory_bodies": ["ENTSO-E", "ACER"],
                "grid_codes": ["IEC 61850", "EN 50549"],
                "time_zones": ["Europe/London", "Europe/Berlin", "Europe/Paris", "Europe/Madrid"]
            },
            Region.ASIA_PACIFIC: {
                "data_retention_days": 1826,  # 5 years
                "encryption_required": True,
                "audit_logging": True,
                "data_sovereignty": True,
                "privacy_framework": "PDPA",
                "currency": "USD",
                "voltage_standard": "IEC",
                "frequency_standard": 50.0,
                "regulatory_bodies": ["AEMO", "EMA"],
                "grid_codes": ["AS/NZS 4777", "TR 25"],
                "time_zones": ["Asia/Tokyo", "Asia/Seoul", "Asia/Singapore", "Australia/Sydney"]
            },
            Region.CHINA: {
                "data_retention_days": 1095,  # 3 years
                "encryption_required": True,
                "audit_logging": True,
                "data_sovereignty": True,
                "privacy_framework": "PIPL",
                "currency": "CNY",
                "voltage_standard": "GB",
                "frequency_standard": 50.0,
                "regulatory_bodies": ["NEA", "SERC"],
                "grid_codes": ["GB/T 19964", "Q/GDW 1755"],
                "time_zones": ["Asia/Shanghai"]
            }
        }
        
        return configs.get(region, configs[Region.NORTH_AMERICA])
        
    def get_compliance_requirements(self) -> Dict[str, Any]:
        """Get compliance requirements for current region."""
        return {
            "data_retention": self.config["data_retention_days"],
            "encryption_required": self.config["encryption_required"],
            "audit_logging": self.config["audit_logging"],
            "data_sovereignty": self.config["data_sovereignty"],
            "privacy_framework": self.config["privacy_framework"]
        }
        
    def get_technical_standards(self) -> Dict[str, Any]:
        """Get technical standards for current region."""
        return {
            "voltage_standard": self.config["voltage_standard"],
            "frequency_standard": self.config["frequency_standard"],
            "regulatory_bodies": self.config["regulatory_bodies"],
            "grid_codes": self.config["grid_codes"]
        }
        
    def validate_compliance(self, data_age_days: int, has_encryption: bool, has_audit_log: bool) -> Dict[str, bool]:
        """Validate compliance with regional requirements."""
        compliance = {
            "data_retention": data_age_days <= self.config["data_retention_days"],
            "encryption": not self.config["encryption_required"] or has_encryption,
            "audit_logging": not self.config["audit_logging"] or has_audit_log
        }
        
        compliance["overall"] = all(compliance.values())
        return compliance


class CurrencyFormatter:
    """Format currency values according to regional settings."""
    
    def __init__(self, region: Region):
        self.region = region
        self.formats = {
            Region.NORTH_AMERICA: {"symbol": "$", "format": "${:,.2f}"},
            Region.EUROPE: {"symbol": "€", "format": "{:,.2f} €"},
            Region.ASIA_PACIFIC: {"symbol": "$", "format": "${:,.2f}"},
            Region.CHINA: {"symbol": "¥", "format": "¥{:,.2f}"},
            Region.SOUTH_AMERICA: {"symbol": "$", "format": "${:,.2f}"},
            Region.MIDDLE_EAST_AFRICA: {"symbol": "$", "format": "${:,.2f}"},
            Region.AUSTRALIA: {"symbol": "A$", "format": "A${:,.2f}"}
        }
        
    def format_currency(self, amount: float) -> str:
        """Format currency amount for region."""
        format_config = self.formats.get(self.region, self.formats[Region.NORTH_AMERICA])
        return format_config["format"].format(amount)
        
    def format_energy_cost(self, energy_mwh: float, price_per_mwh: float) -> str:
        """Format energy cost calculation."""
        total_cost = energy_mwh * price_per_mwh
        return self.format_currency(total_cost)


class UnitConverter:
    """Convert between different unit systems."""
    
    @staticmethod
    def voltage_to_region(voltage_pu: float, region: Region, base_voltage_kv: float = 12.47) -> str:
        """Convert voltage to regional format."""
        voltage_kv = voltage_pu * base_voltage_kv
        
        if region in [Region.NORTH_AMERICA]:
            return f"{voltage_kv:.2f} kV"
        else:
            return f"{voltage_kv:.2f} kV"
            
    @staticmethod
    def power_to_region(power_mw: float, region: Region) -> str:
        """Convert power to regional format."""
        if region == Region.EUROPE:
            return f"{power_mw:.2f} MW"
        elif region == Region.CHINA:
            return f"{power_mw:.2f} MW"
        else:
            return f"{power_mw:.2f} MW"
            
    @staticmethod
    def frequency_to_region(frequency_hz: float, region: Region) -> str:
        """Convert frequency to regional format."""
        return f"{frequency_hz:.2f} Hz"


class GlobalizationManager:
    """Central manager for globalization features."""
    
    def __init__(self, config: Optional[LocalizationConfig] = None):
        if config is None:
            config = LocalizationConfig()
            
        self.config = config
        self.translator = TranslationManager()
        self.regional_config = RegionalConfig(config.region)
        self.currency_formatter = CurrencyFormatter(config.region)
        self.translator.set_language(config.language)
        
    def set_locale(self, language: SupportedLanguage, region: Region) -> None:
        """Set locale for language and region."""
        self.config.language = language
        self.config.region = region
        self.translator.set_language(language)
        self.regional_config = RegionalConfig(region)
        self.currency_formatter = CurrencyFormatter(region)
        
        logger.info(f"Locale set to {language.value}_{region.value}")
        
    def localize_message(self, key: str, **kwargs) -> str:
        """Get localized message."""
        return self.translator.get_translation(key, **kwargs)
        
    def format_performance_metric(self, metric_name: str, value: float, unit: str) -> str:
        """Format performance metric for current locale."""
        if self.config.language == SupportedLanguage.ENGLISH:
            return f"{metric_name}: {value:.3f} {unit}"
        elif self.config.language == SupportedLanguage.SPANISH:
            return f"{metric_name}: {value:.3f} {unit}"
        elif self.config.language == SupportedLanguage.FRENCH:
            return f"{metric_name} : {value:.3f} {unit}"
        elif self.config.language == SupportedLanguage.GERMAN:
            return f"{metric_name}: {value:.3f} {unit}"
        elif self.config.language == SupportedLanguage.JAPANESE:
            return f"{metric_name}: {value:.3f} {unit}"
        elif self.config.language == SupportedLanguage.CHINESE_SIMPLIFIED:
            return f"{metric_name}: {value:.3f} {unit}"
        else:
            return f"{metric_name}: {value:.3f} {unit}"
            
    def get_region_info(self) -> Dict[str, Any]:
        """Get comprehensive region information."""
        return {
            "language": self.config.language.value,
            "region": self.config.region.value,
            "currency": self.config.currency,
            "compliance": self.regional_config.get_compliance_requirements(),
            "technical_standards": self.regional_config.get_technical_standards(),
            "supported_languages": self.translator.get_available_languages()
        }


# Global instance for application-wide use
global_manager = GlobalizationManager()


def set_global_locale(language: str, region: str) -> None:
    """Convenience function to set global locale."""
    try:
        lang_enum = SupportedLanguage(language)
        region_enum = Region(region)
        global_manager.set_locale(lang_enum, region_enum)
    except ValueError as e:
        logger.error(f"Invalid locale parameters: {e}")
        
        
def get_localized_message(key: str, **kwargs) -> str:
    """Convenience function to get localized message."""
    return global_manager.localize_message(key, **kwargs)