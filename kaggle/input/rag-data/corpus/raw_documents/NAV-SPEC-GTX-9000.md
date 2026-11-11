# Naval Gas Turbine Propulsion System Specification GTX-9000 (Rev 4.2)
**Document ID:** NAV-SPEC-GTX-9000 | **Type:** engineering_specification

---

## General System Architecture and Core Capabilities

The GTX-9000 is an aeroderivative marine gas turbine engineered for guided-missile destroyers. The unit delivers a continuous maximum rating of 36.5 MW (48,950 bhp) at an ambient temperature of 15°C (59°F) and sea level barometric pressure. The thermal efficiency at continuous rated power is 41.8%, with an air mass flow rate of 94.2 kg/s. The dual-spool compressor configuration consists of an 8-stage low-pressure compressor (LPC) and a 10-stage high-pressure compressor (HPC) driven by single-stage high-pressure and low-pressure gas generator turbines respectively.

## Operational Speed and Gear Ratio Parameters

The power turbine operating speed ranges from 1,200 RPM at low cruise idle up to a maximum governed speed of 3,600 RPM. The primary reduction gear operates at a reduction ratio of 21.4:1, stepping down the high-speed turbine shaft to a maximum propeller shaft speed of 168.2 RPM. Maximum continuous torque delivered to the shaft flange is 2.07 x 10^6 N·m under full ahead sprint.

## Fuel and Combustion Specifications

The GTX-9000 operates exclusively on NATO F-76 marine diesel or NATO F-44 JP-5 aviation kerosene. Fuel supply pressure delivered to the high-pressure fuel pump manifold must remain continuously within 38.0 to 44.5 bar gauge (551 to 645 psig). Fuel inlet temperature to the dual duplex fuel nozzles must be maintained between 18°C and 45°C. Viscosity at the burner tips must not exceed 6.0 cSt. Automatic fuel cutoff activates when fuel pressure drops below 28.5 bar gauge for longer than 350 milliseconds.

## Turbine Temperature Margins and EGT Limits

Interstage Turbine Temperature (ITT) is monitored continuously by 16 dual-thermocouple probes. Maximum allowable continuous ITT is 845°C (1553°F). Peak allowable transient ITT during rapid acceleration from cruise to sprint is 895°C for a maximum duration of 12 seconds. If ITT exceeds 920°C, the digital engine controller initiates an immediate emergency trip to prevent catastrophic blade creep in the second-stage nozzle guide vanes.

## Lube Oil Cooling & Synthetic Lubricant Constraints

The lubrication system utilizes MIL-PRF-23699 Class C/F synthetic ester lubricant. Nominal lube oil sump capacity is 850 liters. Operating oil pressure at the primary bearings must remain at 4.2 ± 0.3 bar gauge at 3,600 RPM. Normal oil scavenge temperature is 88°C with an emergency high-temperature trip threshold set at 116°C. Lube oil filter differential pressure indicator signals clogging at 1.4 bar differential, and automatic bypass opens at 2.2 bar differential.

## Vibration Spectra and Trip Thresholds

Radial and axial vibration velocity is sampled across five bearing housings (BRG-1 through BRG-5) via triaxial piezoelectric accelerometers. Warning threshold is set at 7.1 mm/s RMS in the 10 Hz to 2,000 Hz broadband band. Automatic turbine shutdown is commanded when overall vibration exceeds 12.7 mm/s RMS for > 1.5 seconds, or if single-order 1X shaft displacement exceeds 48 microns peak-to-peak.

