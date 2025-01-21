#####################################################
######### !!! BU DOSYAYI DEĞIŞTIRMEYIN !!!! #########
#####################################################

import asyncio
from calendar import c
from pathlib import Path
import sys
from typing import Any
from loguru import logger
import pygame

from mocker import ControlPanelThread

# Değişimler
# send_message fonksiyonu kaldırıldı
# is_motor_on fonksiyonu kaldırıldı
# play_sound fonksiyonu eklendi

class Core:
    async def set_motor_angle(self, deg: int) -> None:
        """Servo motorun derecesini ayarlar."""
        _log_out(f"Motor açısı ayarlandı: {deg}°")

    async def get_sound_level(self) -> float:
        """Mikrofonun algıladığı ses seviyesini desibel cinsinden döndürür."""
        return await self._mock_input("Ses seviyesi")

    async def get_temperature(self) -> float:
        """Sıcaklık sensörünün aldığı sıcaklık değerini santigrat cinsinden döndürür."""
        return await self._mock_input("Sıcaklık")

    async def get_humidity(self) -> float:
        """Nem sensörünün aldığı nem değerini yüzde cinsinden döndürür. (0-100)"""
        return await self._mock_input("Nem")

    async def get_ultrasonic_distance(self) -> float:
        """Ultrasonik mesafe sensörü ile ölçülen mesafeyi cm cinsinden döndürür."""
        return await self._mock_input("Mesafe")

    async def get_rain(self) -> float:
        """Yağmur sensörünün algıladığı yağmur miktarını yüzde cinsinden döndürür. (0-100)"""
        return await self._mock_input("Yağmur")

    async def get_light(self) -> float:
        """LDR sensörünün algıladığı ışık miktarını lümen cinsinden döndürür."""
        return await self._mock_input("Işık")

    async def get_gas_amount(self) -> float:
        """Gaz sensörünün algıladığı gaz miktarını ppm cinsinden döndürür."""
        return await self._mock_input("Gaz")

    async def get_proximity(self) -> float:
        """Yakın mesafe sensörünün algıladığı mesafeyi cm cinsinden döndürür."""
        return await self._mock_input("Yakınlık")

    async def get_air_quality(self) -> float:
        """Hava kalitesi sensörünün algıladığı hava kalitesini AQI cinsinden döndürür."""
        return await self._mock_input("Hava Kalitesi")

    async def get_pulse(self) -> float:
        """Nabız sensörünün ölçtüğü nabız değerini BPM cinsinden döndürür."""
        return await self._mock_input("Nabız")

    async def set_state(self, state: str) -> None:
        """Yapay zekanızın durumunu günceller.
        Robotun monitöründe herkesin yapay zekasının durumu gösterilir.
        Bu fonksiyonu çağırdığınızda önceki durumunuz silinir ve yeni durumunuz gösterilir.

        Örnek:
        ```python
        forecast = "Güneşli" // Yapay zekanın tahmin etiği hava durumu
        await set_state("Hava durumu: " + forecast)
        ```
        """
        _log_out(f"Durum güncellendi: {state}")

    async def play_sound(self, sound_path: str | Path) -> None:
        """Belirtilen ses dosyasını çalar."""

        sound_path = Path(sound_path)
        if not sound_path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {sound_path}")

        project_root = Path(__file__).parent
        if sound_path.is_absolute():
            if not sound_path.is_relative_to(project_root):
                raise Exception("Proje klasörünün dışındaki sesleri oynatamazsınız.")

            correct_path = str(sound_path.relative_to(project_root))
            raise Exception(f"C:\\'den itibaren dosya yolu belirtemezsiniz bunun yerine {correct_path!r} yazın.")
        
        sound_path = project_root / sound_path
        if not sound_path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {sound_path}")

        _log_out(f"Ses çalınıyor: {sound_path}")


        pygame.mixer.music.load(str(sound_path))
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.05)



    def __init__(self) -> None:
        self._control_panel_thread = ControlPanelThread()
        self._control_panel_thread.start()
        pygame.mixer.init()

    async def _mock_input(self, label: str) -> Any:
        await asyncio.sleep(0.2)
        value = self._control_panel_thread.get_value(label)
        _log_in(f"{label}: {value!r}")
        return value


# Kendi bilgisayarınızda test edebilmeniz için için ekranda güzel bir çıktı oluşturduk
logger.level("INPUT", no=1, icon="❯❯")
logger.level("OUTPUT", no=1, icon="❮❮", color="<blue>")
logger.add(
    sys.stderr,
    colorize=True,
    level=1,
    filter="core",
    format="<green>{time:HH:mm:ss.SSS}</green> <level>{level.icon} {message}</level>",
)


def _log_in(s: str):
    logger.log("INPUT", s)


def _log_out(s: str):
    logger.log("OUTPUT", s)
