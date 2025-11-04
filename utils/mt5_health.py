import MetaTrader5 as mt5
import time

def ensure_mt5_ready(retries=3, sleep_s=2):
    """Ponawia initialize i weryfikuje, że terminal + konto są gotowe."""
    for i in range(retries):
        if mt5.initialize():
            ti, ai = mt5.terminal_info(), mt5.account_info()
            if ti and ai and getattr(ai, "login", 0):
                return True
            mt5.shutdown()
        time.sleep(sleep_s * (i + 1))
    return False
