# SunCET Realtime Telemetry Display

This package reads realtime telemetry from a TCP client connection, extracts
CCSDS packets, decodes them with the existing CTDB-generated decoders, and
serves a local browser dashboard over HTTP/SSE. It accepts the current
ASM-plus-CCSDS stream and can still handle the earlier UHF wrappers when enabled
in `config.ini`.

## Run

From the repository root:

```bash
python -m suncet_processing_pipeline.realtime_display.main
```

Default settings live in:

```text
suncet_processing_pipeline/realtime_display/config.ini
```

The TCP endpoint is configured in `config.ini`. The dashboard defaults to:

```text
http://127.0.0.1:8050/
```

Drag horizontally across any sparkline to zoom all plots to the selected onboard
time range. Double-click a plot, or use **Reset Zoom**, to return to the full
rolling history.

## Onboard Time

The top-left **Onboard UTC** value and plot x-axis times come from the first
`ccsdsSecHeader2_sec*` coarse time in each decoded packet whenever that secondary
header is present. The realtime config treats that coarse time as FSW J2000
seconds since midnight UTC and adds post-J2000 leap seconds before display:

```ini
[time]
j2000_epoch_utc = 2000-01-01T00:00:00Z
add_post_j2000_leap_seconds = true
```

## Replay A Capture

```bash
python -m suncet_processing_pipeline.realtime_display.main --replay /path/to/capture.bin --replay-delay 0.05
```

Replay uses the same parser, segmentation reassembly, checksum option, CTDB
decode path, and dashboard state as live TCP mode.

## Demo Dashboard

```bash
python -m suncet_processing_pipeline.realtime_display.main --demo
```

Demo mode generates synthetic temperature and voltage values so the browser
display can be checked without hardware or captured bytes.

## Framing Assumptions

The default parser path is now standard CCSDS attached sync marker (ASM) followed
by a normal CCSDS packet:

```text
0x1ACFFC1D + CCSDS primary header + CCSDS secondary header + user data
```

The parser also accepts these forms when enabled in `config.ini`:

1. A fixed upstream frame prefix plus direct CCSDS packet.
2. AX.25 header plus direct CCSDS packet.
3. AX.25 header plus segmented realtime header plus CCSDS packet chunks.
4. Direct CCSDS packet when the ASM or wrapper has already been stripped.
5. Segmented realtime header plus chunks when the bridge has stripped AX.25.

The current TCP tap has been observed using a 16-byte prefix beginning with
`0x1BADCafe`, configured as:

```text
0x1BADCafe + 12 prefix bytes + CCSDS primary header + CCSDS secondary header + user data
```

For the current alternate tap point, the parser can also recover direct CCSDS
packets after a fixed or variable upstream prefix by scanning for a valid CCSDS
primary header. Explicit sync markers and fixed prefixes are preferred over that
loose scan so payload bytes that resemble CCSDS headers do not block later real
packets.

The AX.25 `ctrl` and `pid` bytes are configurable and default to the FSW fixed
values `0x03` and `0xF0`. The parser uses them as a sync aid, but the underlying
decode path does not depend on them.

Segmented packets are reassembled with big-endian header fields and the current
flag convention:

```text
start = 1
middle = 0
end = 2
```

## Checksum Mode

Packet checksums are available but disabled by default:

```ini
[decode]
require_packet_checksums = false
drop_failed_checksums = false
```

Set `require_packet_checksums = true` to mark packets with checksum status. Set
`drop_failed_checksums = true` only when you want failed packets excluded from the
display state.
