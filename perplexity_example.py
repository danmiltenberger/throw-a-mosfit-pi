# https://www.perplexity.ai/search/d4c7c143-d2bc-43cd-b8b5-1b450ea3a9b7

from smbus3 import SMBus, i2c_msg


def send_text_i2c(bus_number: int, arduino_address: int, text: str,
                  encoding: str = "utf-8") -> None:
    """
    Send a text string to an Arduino configured as an I2C slave.

    Parameters
    ----------
    bus_number : int
        I2C bus number on the host, commonly 1 on Raspberry Pi/Linux SBCs.
    arduino_address : int
        7-bit I2C address of the Arduino, e.g. 0x08.
    text : str
        Text to transmit.
    encoding : str
        Character encoding used to convert text into bytes.

    Raises
    ------
    ValueError
        If the encoded message exceeds the Arduino I2C receive buffer size.
    """
    data = text.encode(encoding)

    # Classic Arduino Wire implementations commonly have a 32-byte buffer.
    # Reserve nothing here because this sends raw payload bytes only.
    if len(data) > 32:
        raise ValueError(
            f"Message is {len(data)} bytes; limit is 32 bytes for a typical "
            "Arduino Wire receive buffer. Send it in chunks instead."
        )

    with SMBus(bus_number) as bus:
        message = i2c_msg.write(arduino_address, list(data))
        bus.i2c_rdwr(message)


send_text_i2c(
    bus_number=1,
    arduino_address=0x08,
    text="Hello, Arduino!"
)