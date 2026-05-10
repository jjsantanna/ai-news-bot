"""Seed an ISO 27001:2022 Annex A controls xlsx.

Writes the 93 publicly-known control ids and titles (the names of the controls
are factual and not copyrightable). The `description` column is left blank
deliberately — you must paraphrase the normative text yourself; we do not
redistribute the ISO standard's clause language.

Run:
    uv run python scripts/seed_iso_27001.py --out standards/iso_27001_2022_annex_a.xlsx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from compliance_extractor import schemas
from compliance_extractor.io.controls_xlsx import write_controls

# (control_id, theme, title)
ANNEX_A_CONTROLS: list[tuple[str, str, str]] = [
    ("A.5.1", "Organizational", "Policies for information security"),
    ("A.5.2", "Organizational", "Information security roles and responsibilities"),
    ("A.5.3", "Organizational", "Segregation of duties"),
    ("A.5.4", "Organizational", "Management responsibilities"),
    ("A.5.5", "Organizational", "Contact with authorities"),
    ("A.5.6", "Organizational", "Contact with special interest groups"),
    ("A.5.7", "Organizational", "Threat intelligence"),
    ("A.5.8", "Organizational", "Information security in project management"),
    ("A.5.9", "Organizational", "Inventory of information and other associated assets"),
    ("A.5.10", "Organizational", "Acceptable use of information and other associated assets"),
    ("A.5.11", "Organizational", "Return of assets"),
    ("A.5.12", "Organizational", "Classification of information"),
    ("A.5.13", "Organizational", "Labelling of information"),
    ("A.5.14", "Organizational", "Information transfer"),
    ("A.5.15", "Organizational", "Access control"),
    ("A.5.16", "Organizational", "Identity management"),
    ("A.5.17", "Organizational", "Authentication information"),
    ("A.5.18", "Organizational", "Access rights"),
    ("A.5.19", "Organizational", "Information security in supplier relationships"),
    ("A.5.20", "Organizational", "Addressing information security within supplier agreements"),
    ("A.5.21", "Organizational", "Managing information security in the ICT supply chain"),
    (
        "A.5.22",
        "Organizational",
        "Monitoring, review and change management of supplier services",
    ),
    ("A.5.23", "Organizational", "Information security for use of cloud services"),
    ("A.5.24", "Organizational", "Information security incident management planning and preparation"),
    ("A.5.25", "Organizational", "Assessment and decision on information security events"),
    ("A.5.26", "Organizational", "Response to information security incidents"),
    ("A.5.27", "Organizational", "Learning from information security incidents"),
    ("A.5.28", "Organizational", "Collection of evidence"),
    ("A.5.29", "Organizational", "Information security during disruption"),
    ("A.5.30", "Organizational", "ICT readiness for business continuity"),
    ("A.5.31", "Organizational", "Legal, statutory, regulatory and contractual requirements"),
    ("A.5.32", "Organizational", "Intellectual property rights"),
    ("A.5.33", "Organizational", "Protection of records"),
    ("A.5.34", "Organizational", "Privacy and protection of PII"),
    ("A.5.35", "Organizational", "Independent review of information security"),
    (
        "A.5.36",
        "Organizational",
        "Compliance with policies, rules and standards for information security",
    ),
    ("A.5.37", "Organizational", "Documented operating procedures"),
    ("A.6.1", "People", "Screening"),
    ("A.6.2", "People", "Terms and conditions of employment"),
    ("A.6.3", "People", "Information security awareness, education and training"),
    ("A.6.4", "People", "Disciplinary process"),
    ("A.6.5", "People", "Responsibilities after termination or change of employment"),
    ("A.6.6", "People", "Confidentiality or non-disclosure agreements"),
    ("A.6.7", "People", "Remote working"),
    ("A.6.8", "People", "Information security event reporting"),
    ("A.7.1", "Physical", "Physical security perimeters"),
    ("A.7.2", "Physical", "Physical entry"),
    ("A.7.3", "Physical", "Securing offices, rooms and facilities"),
    ("A.7.4", "Physical", "Physical security monitoring"),
    ("A.7.5", "Physical", "Protecting against physical and environmental threats"),
    ("A.7.6", "Physical", "Working in secure areas"),
    ("A.7.7", "Physical", "Clear desk and clear screen"),
    ("A.7.8", "Physical", "Equipment siting and protection"),
    ("A.7.9", "Physical", "Security of assets off-premises"),
    ("A.7.10", "Physical", "Storage media"),
    ("A.7.11", "Physical", "Supporting utilities"),
    ("A.7.12", "Physical", "Cabling security"),
    ("A.7.13", "Physical", "Equipment maintenance"),
    ("A.7.14", "Physical", "Secure disposal or re-use of equipment"),
    ("A.8.1", "Technological", "User end point devices"),
    ("A.8.2", "Technological", "Privileged access rights"),
    ("A.8.3", "Technological", "Information access restriction"),
    ("A.8.4", "Technological", "Access to source code"),
    ("A.8.5", "Technological", "Secure authentication"),
    ("A.8.6", "Technological", "Capacity management"),
    ("A.8.7", "Technological", "Protection against malware"),
    ("A.8.8", "Technological", "Management of technical vulnerabilities"),
    ("A.8.9", "Technological", "Configuration management"),
    ("A.8.10", "Technological", "Information deletion"),
    ("A.8.11", "Technological", "Data masking"),
    ("A.8.12", "Technological", "Data leakage prevention"),
    ("A.8.13", "Technological", "Information backup"),
    ("A.8.14", "Technological", "Redundancy of information processing facilities"),
    ("A.8.15", "Technological", "Logging"),
    ("A.8.16", "Technological", "Monitoring activities"),
    ("A.8.17", "Technological", "Clock synchronization"),
    ("A.8.18", "Technological", "Use of privileged utility programs"),
    ("A.8.19", "Technological", "Installation of software on operational systems"),
    ("A.8.20", "Technological", "Networks security"),
    ("A.8.21", "Technological", "Security of network services"),
    ("A.8.22", "Technological", "Segregation of networks"),
    ("A.8.23", "Technological", "Web filtering"),
    ("A.8.24", "Technological", "Use of cryptography"),
    ("A.8.25", "Technological", "Secure development life cycle"),
    ("A.8.26", "Technological", "Application security requirements"),
    ("A.8.27", "Technological", "Secure system architecture and engineering principles"),
    ("A.8.28", "Technological", "Secure coding"),
    ("A.8.29", "Technological", "Security testing in development and acceptance"),
    ("A.8.30", "Technological", "Outsourced development"),
    ("A.8.31", "Technological", "Separation of development, test and production environments"),
    ("A.8.32", "Technological", "Change management"),
    ("A.8.33", "Technological", "Test information"),
    ("A.8.34", "Technological", "Protection of information systems during audit testing"),
]


def build_dataframe(standard: str = "ISO 27001:2022") -> pd.DataFrame:
    rows = [
        {
            schemas.ControlsCols.CONTROL_ID: cid,
            schemas.ControlsCols.TITLE: title,
            schemas.ControlsCols.DESCRIPTION: "",
            schemas.ControlsCols.THEME: theme,
            schemas.ControlsCols.STANDARD: standard,
        }
        for cid, theme, title in ANNEX_A_CONTROLS
    ]
    df = pd.DataFrame(rows)
    return schemas.cast(df, schemas.CONTROLS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--standard", default="ISO 27001:2022")
    args = parser.parse_args()
    df = build_dataframe(standard=args.standard)
    write_controls(df, args.out)
    print(f"Wrote {args.out} ({len(df)} controls)")


if __name__ == "__main__":
    main()
