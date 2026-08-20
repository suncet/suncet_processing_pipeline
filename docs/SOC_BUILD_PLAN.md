# SunCET SOC Jetson Build and Operations Plan

Last updated: 2026-08-20

## Purpose

This document tracks the build of `suncet-soc`, an NVIDIA Jetson AGX Orin used as
the SunCET Science Operations Center processing node. The initial operating model
is manual processing after a satellite pass. Automated ingest and processing will
be added after the pipeline and its data sources are stable.

## Current system

- Hostname: `suncet-soc`
- Administrator account: `james`
- Operating system: Ubuntu 24.04.4 LTS, ARM64
- NVIDIA platform: L4T R39.2.1, Jetson AGX Orin
- Current storage: internal eMMC; a Samsung 990 PRO 2 TB NVMe is scheduled for
  installation
- Current network: `JHUAPL-Staff` Wi-Fi using DHCP
- Remote administration: SSH through the APL VPN
- Repository: <https://github.com/suncet/suncet_processing_pipeline>

Remote access was validated on 2026-08-20 from a Mac connected to a phone
hotspot and the APL VPN. `ssh james@suncet-soc` succeeded, and routing to the
Jetson used a VPN `utun` interface. A static IP is therefore not required at
present. The Jetson must remain powered, connected to Staff Wi-Fi, and reachable
under its hostname.

## Decisions

- Work under the `james` account, using passwordless SSH key authentication.
- Use Git `main` for the small development team; record the Git revision for
  every processing run.
- Use Miniforge/Mamba and the repository environment definitions.
- Support both ARM and Intel hosts. Python 3.12 and the current Python 3.14
  release are tested; the environment is not tied to the Mac's ARM architecture.
- Keep CUDA and JetPack components at the operating-system level rather than in
  the portable Python environment.
- Use `imagecodecs` for JPEG-LS decompression.
- Keep the operating system, boot support, SSH access, source checkout, and a
  small reproducible production environment on eMMC. Use NVMe for SunCET data,
  products, processing scratch space, large caches, optional development
  environments, and container storage. This is the intended permanent layout,
  not merely a migration stage.
- Preserve the local `suncet_data` directory organization on the Jetson. After
  NVMe installation, point `suncet_data` at the NVMe data root so the pipeline
  does not depend on the physical storage location.
- Keep private CTDB content outside `suncet_data`, which is publicly exposed and
  synchronized on development systems. Define the separate host-local
  `suncet_ctdb` root on every system. On the Jetson it will be a local directory,
  not a Box dependency, and must never be nested inside the public data root.
- Use ARM64 `rclone` with explicit one-way `copy` operations for the initial
  Dropbox integration. Dropbox is a replication/publication path, not the
  processing queue, sole backup, or authority for in-progress files. Do not use
  the unsupported native Dropbox Linux client or begin with bidirectional sync.
- Begin with manual data transfer and processing. Do not add unattended jobs
  until the manual workflow is reliable and observable.
- Do not expose SSH directly to the public Internet. Use the APL VPN or another
  APL-approved management path.

## Security and APL network disposition

IT advised that this ARM64 Jetson should use an InfoSec policy exception to
"Extend APLNIS Connectivity for EOL / Unsupported / Retired OS" if APLNIS
connectivity is required. The current architecture does not require the Jetson
itself to join APLNIS: it remains on `JHUAPL-Staff`, while administrators reach
it from an APL-managed Mac through the APL VPN. Offsite VPN/SSH access has been
validated.

The selected posture is therefore:

- Proceed with the current NVIDIA Ubuntu/L4T installation on Staff Wi-Fi.
- Do not reimage with the APL Golden Image or install LCMS/BigFix.
- Do not request the APLNIS unsupported-platform exception unless APLNIS
  connectivity later becomes an operational requirement.
- Do not connect the Jetson to APLNIS Ethernet without first obtaining the
  required approval.
- Keep the system manually patched, restrict remote administration to SSH keys
  over APL-approved network paths, and do not expose public inbound services.
- Revisit the network/security design before storing controlled data, adding
  mission credentials, accepting inbound production services, or publishing
  products from the Jetson.

This interpretation is based on IT's response in `INC0541924`: the exception is
for extending APLNIS connectivity to the unsupported platform. A short ticket
reply confirming that no exception is needed while the Jetson stays off APLNIS
would provide a useful written record, but it does not block reversible
user-space environment work.

## Roadmap and status

### 0. Put development under reproducible version control — complete

- Important local pipeline work was committed and pushed to `main`.
- GitHub Actions passes on Python 3.12 and Python 3.14.
- Portable environment modernization is in commit `95b56bc`.
- Processing-run provenance is in commit `a24cd1d`.

### 1. Record a read-only Jetson baseline — complete

The baseline captured OS, kernel, L4T, storage, memory, CPU, power mode,
temperature, time synchronization, NVIDIA driver/CUDA compatibility, installed
development tools, containers, and LCMS/BigFix status. Important findings:

- The NVIDIA driver reports CUDA 13.2 compatibility, but the complete CUDA
  toolkit and `nvcc` are not installed.
- Docker, containerd, and NVIDIA Container Toolkit are installed.
- Miniforge/Mamba is not installed.
- eMMC has adequate short-term capacity but is not the intended production data
  volume.
- Time synchronization is active through `systemd-timesyncd`.

### 2. Make the Python environment portable and Jetson-ready — complete

Completed in the repository:

- `environment.yml` defines the production/runtime environment.
- `environment-dev.yml` contains development, test, and notebook tools.
- `environment-radio.yml` isolates optional GNU Radio dependencies.
- `conda-lock.yml` locks tested packages for `linux-aarch64`, `linux-64`,
  `osx-arm64`, and `osx-64`.
- Lock solves succeeded for all four target platforms.
- The test suite passes under Python 3.12 and 3.14.
- Lossless 16-bit JPEG-LS round-trip behavior was validated with `imagecodecs`.
- Managed paths now use one required absolute `suncet_data` root; workstation
  paths and repository-relative processing storage have been removed from code.
- Private CTDBs use the separate required `suncet_ctdb` root. The pipeline
  rejects overlapping public-data and CTDB roots, and provenance manifests hide
  the host-specific private root.
- Regression tests scan Python, configuration, and notebook sources for
  workstation paths or direct legacy environment lookups.

Completed on the Jetson on 2026-08-20:

- Cloned `main` at commit `f61c75f9ccbccea984765a98ed7db24637e2a2ae`
  into `/home/james/src/suncet_processing_pipeline`; the checkout was clean at
  the validation gate.
- Checksum-verified and installed ARM64 Miniforge 26.3.2-3 under
  `/home/james/miniforge3`. Conda base activation is disabled by default.
- Created the production `suncet` environment from the repository's exact
  `linux-aarch64` lock and installed the checkout in editable mode without
  resolving its dependencies again.
- Verified native ARM64 Python 3.14.7, `imagecodecs` 2026.8.16, the complete
  science-package import set, and `pip check` with no broken requirements.
- Created a cloned `suncet-test` validation environment with only pytest and
  coverage tooling added. All 59 repository tests passed in 8.12 seconds.
- Verified lossless 16-bit JPEG-LS encode/decode with representative image data.
  An incompressible random-image encoder stress test also passed when given an
  explicit output buffer. The default encoder buffer can be too small for that
  artificial worst case; operational SunCET code consumes JPEG-LS by decoding,
  so this is recorded as a test-tool behavior rather than a processing blocker.
- Created `/home/james/suncet_ctdb` with mode `700`, exported it as
  `suncet_ctdb` from `.bashrc`, and retained a pre-change Bash configuration
  backup. No CTDB contents have been transferred yet.
- Intentionally left `suncet_data` undefined until the NVMe is mounted at its
  permanent path. Validation used a temporary data root, preventing an interim
  eMMC location from becoming operational by accident.
- Miniforge, both environments, and the shared package cache use about 4.7 GB;
  approximately 40 GB remains free on eMMC.

A representative Level 0.5 dataset comparison remains in Roadmap Step 6 after
the NVMe data root and required CTDB contents are available.

### 3. Improve processing-run reproducibility — complete

The pipeline writes an atomic JSON manifest for each processing run under the
data directory's `processing_manifests` folder. It records timing, success or
failure, sanitized arguments and configuration, Git state, environment and
package details, input and output checksums, and exception details. Successful
and intentionally failed runs have been tested.

### 4. Choose the initial APL network posture — complete

Keep the Jetson on Staff Wi-Fi and off APLNIS. IT directed unsupported systems
that need APLNIS connectivity to the InfoSec exception process; this build does
not currently need that extension. Reopen this decision before moving to
Ethernet/APLNIS or expanding the node's data and service exposure.

### 5. Establish the short-term data layout and manual ingest — pending

- After NVMe installation, create the permanent data root and export it as
  `suncet_data` for shell and non-interactive processing contexts.
- Reproduce the directory organization currently rooted at
  `/Users/masonjp2/Dropbox/suncet_dropbox/9000 Processing/data/` on the Mac.
- Install the ARM64 build of `rclone` and configure a Dropbox remote using a
  dedicated identity that can access only SunCET public data, if that account
  arrangement is available. Avoid placing a token for the owner's full personal
  Dropbox account on the Jetson. Store the rclone configuration with permissions
  restricted to `james`.
- Use explicit one-way `rclone copy` commands rather than `sync` or `bisync`:
  copy reference inputs from Dropbox to the Jetson, and copy finalized products
  and processing manifests from the Jetson to Dropbox. Because `copy` does not
  delete destination files, remote or local deletion will not be propagated by
  the initial workflow.
- Maintain a reviewed rclone filter file that includes only intended public
  subtrees. Exclude scratch space, caches, temporary/partial products, package
  stores, and all private material. `suncet_ctdb` is outside `suncet_data` and
  must never be an rclone source or destination.
- Write products outside the exported Dropbox subtree while they are incomplete,
  then move completed products atomically into their publication location.
- Begin with manual `--dry-run` and logged copy operations. Verify representative
  transfers in both directions before considering a systemd timer. Any later
  move to `sync` or `bisync` requires a separate review of ownership, conflicts,
  deletion propagation, and recovery behavior.
- Document a manual, resumable pull from
  <https://lasp.colorado.edu/data/store/suncet/>.
- Verify downloaded files with server metadata or checksums where available.
- Define retention thresholds so the eMMC cannot be filled by an ingest.
- Defer AWS/Leaf Space credentials and LASP write-back until their operational
  interfaces and authorization are known.

### 6. Validate the end-to-end manual processing workflow — pending

- Process a known dataset from ingest through the currently implemented levels.
- Compare Jetson products and manifests against a known-good Mac run.
- Measure elapsed time, peak memory, disk growth, and failure behavior.
- Write a concise operator runbook covering input discovery, processing,
  product review, retry, and recovery.

### 7. Add NVMe storage and establish the permanent storage split — hardware ordered

- Install the Samsung 990 PRO 2 TB M.2 2280 NVMe with suitable thermal contact.
- Partition it as ext4 and mount it by UUID at a stable, documented location.
- Allow the operating system to boot into a maintainable state if the NVMe is
  absent, but require the mount before any processing service can start.
- Move any temporary data with a metadata-preserving copy, verify the copy, and
  only then change `suncet_data`; retain the source until validation succeeds.
- Keep the OS, SSH recovery path, repository, and minimal production runtime on
  eMMC. Direct high-volume and write-heavy storage to NVMe.
- Review Mamba package caches, optional environments, and Docker's data root
  before they grow; configure large stores directly on NVMe rather than copying
  installed environments afterward.
- Define backup, retention, filesystem monitoring, and recovery procedures.
- Keep code and environment reconstruction possible from Git and lock files.

### 8. Install and validate GPU compute support — pending operational need

- Inventory NVIDIA packages before making changes.
- Select the JetPack/CUDA packages compatible with L4T R39.2.1.
- Install only the required system components and validate with a small CUDA
  workload.
- Benchmark CPU and GPU implementations of PSF deconvolution when the
  GPU-enabled implementation is ready.
- Record Jetson power mode, clocks, temperature, throughput, memory, and energy
  per product for future flight-hardware suitability studies.

### 9. Automate post-pass operations — future

After manual operations are dependable:

- Detect newly available LASP or Leaf Space inputs without duplicating work.
- Stage downloads atomically and validate completeness before processing.
- Run processing under a dedicated service or timer with an explicit locked
  environment and data root.
- Preserve manifests and logs; alert on failed or stale runs.
- Make retries idempotent and prevent concurrent processing of the same input.
- Coordinate maintenance and reboot windows with mission operations.

### 10. Consider product publication — future and authorization-dependent

Public hosting on APL infrastructure or product write-back to LASP requires a
separate architecture and permission review. It is not part of the initial SOC
build.

## Immediate next action

Install and validate the NVMe in Roadmap Step 7, then establish the permanent
`suncet_data` root and manual LASP/Dropbox transfer workflow in Roadmap Step 5.
After the required CTDB versions and a known dataset are available, execute the
representative Level 0.5 comparison in Roadmap Step 6. No data-ingest automation
should begin before those manual validation gates pass.

## Definition of an initial operational SOC

The initial build is ready for routine manual use when:

- VPN/SSH administration works from an offsite network.
- The Jetson remains off APLNIS unless the required exception is approved.
- The locked ARM64 environment installs reproducibly on the Jetson.
- Unit tests and JPEG-LS validation pass on the Jetson.
- A known dataset produces products consistent with a known-good Mac run.
- Every run emits a complete provenance manifest.
- Data paths, free-space thresholds, and manual recovery steps are documented.
