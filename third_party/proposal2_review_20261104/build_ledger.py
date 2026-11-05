"""Render acquisition registry and explicitly recorded reading scopes, separately."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parent
DOC = ROOT.parents[1] / 'docs' / 'proposal2'
PROGRESS = {
 'molingo': {'level':'D+C','note':'evidence/M01_molingo.md', 'scope':'§§3–5; supplement §§6–7; Tables1–3,5–10; scoped code audit', 'venue':'CVPR2026 — official CVF index'},
 'frankenmotion': {'level':'D-main','note':'evidence/M02_frankenmotion.md', 'scope':'main §§3–6, Tables1–3; supplementary/code/video chưa đọc', 'venue':'CVPR2026 — official CVF index'},
 'lgtok': {'level':'D','note':'evidence/M03_lgtok.md', 'scope':'§§3–4; AppB/D/E/G/I/K; chưa đọc kỹ AppC/KIT-ML', 'venue':'ECCV2026 — author-reported acceptance, proceedings chưa đối chiếu'},
 'eventt2m': {'level':'D','note':'evidence/M04_event_t2m.md', 'scope':'§§3–6; AppA.1,5–7,9,11,14–18; Table15 chưa đọc mọi hàng', 'venue':'ICLR2026 — official proceedings entry verified'},
 'omnimotion': {'level':'V','scope':'abstract gốc qua search; chưa đọc method', 'venue':'preprint; OpenReview decision chưa xác minh'},
 'mobius': {'level':'M','scope':'abstract/metadata; method§3.1–3.3 đến PhaseII; chưa đọc experiments/ablations', 'venue':'preprint 03/11/2026'},
 't2mbench': {'level':'V','scope':'arXiv abstract qua search; chưa đọc method', 'venue':'preprint; venue chưa xác minh'},
 'open_t2m': {'level':'M','scope':'§§3–4 và5.1: data curation/2D-PRQ/instruction tuning; chưa đọc kết quả/ablations','venue':'CVPR2026 — official CVF index'},
 'mardm': {'level':'D','note':'evidence/M05_mardm.md','scope':'main §§2–4, Tables1–6; AppA/C/D/G, TablesA1–A6; chưa code','venue':'CVPR2025 — official CVF metadata'},
 'gesture_token_semantics': {'level':'D','note':'evidence/M06_gesture_token_semantics.md','scope':'main+appendix, Tables1–5; chưa code/temporal probes','venue':'preprint; venue chưa xác minh'},
 'umf': {'level':'M','scope':'abstract, §§1–4.2.2 đến Eq14; number-free nói về số agents, không phải duration; chưa đọc experiments/ablations','venue':'CVPR2026 — official CVF metadata'},
 'momask': {'level':'D','note':'evidence/M07_momask.md','scope':'main §§3–5, Tables1–2 và limitations; chưa code/video','venue':'CVPR2024 — official CVF index'},
 'human_tomato': {'level':'D','note':'evidence/M08_human_tomato.md','scope':'arXivv1 §§2–3/AppB/C; published ICML2024 §§3–4, AppB/C/K, scoped version comparison; PDFpreflight UNAVAILABLE','venue':'ICML2024 — official PMLR235/lu24b'},
 'emage': {'level':'D','note':'evidence/M09_emage.md','scope':'§§3–5, AppB statistics, C/D/F; chưa code/video','venue':'CVPR2024 — official CVF metadata'},
 'party': {'level':'D','note':'evidence/M10_party.md','scope':'main §§3–4; AppA,E.1–E.4; chưa toàn bộ metric supplement/userstudy/code','venue':'CVPR2026 — official CVF metadata'},
 'mdm': {'level':'D','note':'evidence/M11_generator_foundations.md','scope':'§§3–6; text-to-motion setup, Tables1–4, geometric ablations/action task; chưa code/video','venue':'ICLR2023 — official OpenReview published PDF verified'},
 'mld': {'level':'D','note':'evidence/M11_generator_foundations.md','scope':'§3,§4.1–4.2,§5/Tables4–6, AppB.3/B.4,C,H.3; chưa code','venue':'CVPR2023 — official CVF entry verified'},
 't2mgpt': {'level':'D','note':'evidence/M11_generator_foundations.md','scope':'§§3–4.1; training/discussion, Tables3–7/AppA–D; chưa code/video','venue':'CVPR2023 — official CVF entry verified'},
 'motionstreamer': {'level':'D','note':'evidence/M12_motionstreamer.md','scope':'§§3–4, Tables1–4, AppA/E/F; chưa code/đầy đủ sweeps/video','venue':'ICCV2025 — official CVF metadata'},
 'discord': {'level':'D','note':'evidence/M13_discord.md','scope':'§§3–4, Tables1–6, AppA,C.4; chưa code/đầy đủ auxiliary-task supplement','venue':'ICCV2025 — official CVF metadata'},
 'stride': {'level':'D','note':'evidence/M14_stride.md','scope':'§§2–4, AppA–C,E; chưa toàn bộ adapted-VLM tests/code','venue':'preprint; venue chưa xác minh'},
 'handx': {'level':'D','note':'evidence/M15_handx.md','scope':'main§§3–6; AppA.2–5,C,E.1–3,G; chưa toàn bộ fitting/control/userstudy/code','venue':'CVPR2026 — official CVF metadata'},
 'flowmdm': {'level':'D','note':'evidence/M16_temporal_planning.md','scope':'§§3–4, Tables1–4, supplementaryA/B; chưa đầy đủ sweeps/video/code','venue':'CVPR2024 — official CVF entry verified'},
 'plan_dont_pose': {'level':'D','note':'evidence/M16_temporal_planning.md','scope':'§§3–4, Tables1–6, AppA; chưa audit propositions/code/simulator','venue':'preprint; venue chưa xác minh'},
 'mar': {'level':'D','note':'evidence/M17_cross_domain_transfer.md','scope':'§§3–5, Tables1–2, ablations, AppA,D.3; chưa code','venue':'NeurIPS2024 — official proceedings verified'},
 'f5tts': {'level':'D','note':'evidence/M17_cross_domain_transfer.md','scope':'§§3–5 vàAppB.2; chưa mọi solver/data-scale/userstudy table/code','venue':'ACL2025 — official Anthology entry verified'},
 'glowtts': {'level':'D','note':'evidence/M17_cross_domain_transfer.md','scope':'§§3–5.3 vàAppB.1; chưa voice conversion/code','venue':'NeurIPS2020 — official proceedings verified'},
 'act': {'level':'D','note':'evidence/M17_cross_domain_transfer.md','scope':'§§IV–VII, AppF; chưa từng hyperparameter table/code','venue':'RSS2023 — official proceedings verified'},
 'soke': {'level':'D+C','note':'evidence/S01_soke.md','scope':'main+appendices; static audit representation/grouping/default config và DTW/alignment; chưa generation/BT đầy đủ','venue':'ICCV2025 — official CVF entry verified'},
 'madis': {'level':'D+C','note':'evidence/S02_madis.md','scope':'main method/setup/ablations/humanstudy; static generation/loss/SiBLEU audit + synthetic metric diagnostic; chưa full SiCLIP/supplement','venue':'preprint; peer-review status chưa xác minh'},
}

# Explicitly recorded scopes from the SLG and targeted novelty review.
PROGRESS.update({'signspark': {'level': 'D+C', 'note': 'evidence/S03_signspark.md', 'scope': 'main v4 method/setup/Tables 1–8; static training/sampling/length/release audit; no separate supplement', 'venue': 'ECCV2026 — author-reported acceptance; proceedings entry chưa đối chiếu'}, 'signer': {'level': 'D', 'note': 'evidence/S04_signer.md', 'scope': 'main v2 method/setup/results/ablations/limitations + official four-page supplement; chưa code', 'venue': 'ECCV2026 — author-reported acceptance; proceedings entry chưa đối chiếu'}, 'nsa': {'level': 'D', 'note': 'evidence/S05_nsa.md', 'scope': 'fitting/generator/setup/Tables 1–3/human study; chưa code hoặc toàn supplementary', 'venue': 'CVPR2024 — official CVF paper'}, 'signrr': {'level': 'D', 'note': 'evidence/S06_signrr.md', 'scope': 'main §§3–5/Tables 2–6 + supplementary A–G; chưa code', 'venue': 'preprint; venue chưa xác minh'}, 'signflip': {'level': 'D', 'note': 'evidence/S07_signflip.md', 'scope': '§§3–4, limitations, App A/B/D/E; chưa video/toàn questionnaire/code', 'venue': 'preprint; venue chưa xác minh'}, 'fgdm': {'level': 'M', 'note': 'evidence/S08_fgdm.md', 'scope': '§3.1/forward diffusion/focal overview, §3.3 SCG, §4.2 narrative; chưa full ASGCN/ablations/code', 'venue': 'CVPR2026 — official CVF PDF'}, 'blanc': {'level': 'D', 'note': 'evidence/S09_blanc.md', 'scope': '§§3–5/Tables 1–6, App B/C/D.1–D.5; chưa toàn phần cuối appendix/video/code', 'venue': 'preprint 10/2026; project under review'}, 'handmdm': {'level': 'V+intro', 'note': 'evidence/S10_scope_checks.md', 'scope': 'abstract và introduction/task/data source; chưa method/results', 'venue': 'CVPR2026 — official CVF metadata acquired'}, 'ksl_nms': {'level': 'M', 'note': 'evidence/S10_scope_checks.md', 'scope': '§3.3 và §§4.1–4.4; chưa deep results/ablations/code', 'venue': 'Sensors2026 — DOI/publisher/PubMed verified'}, 'nmm_asl': {'level': 'M', 'note': 'evidence/S10_scope_checks.md', 'scope': 'abstract và introduction, timing/linguistic scope only; chưa methods/results deep', 'venue': 'PLOS ONE2014 — DOI/PMC verified'}, 'transition': {'level': 'M', 'note': 'evidence/S10_scope_checks.md', 'scope': '§3 và chọn lọc setup/results tables; completion regime; chưa deep supplementary/code', 'venue': 'CVPR2025 — official CVF metadata acquired'}, 'segmo': {'level': 'D', 'note': 'evidence/M18_segmo.md', 'scope': '§§3.1–3.5, setup/implementation, §4.2–4.3/Table 3/limitations; chưa supplementary/code', 'venue': 'WACV2026 — official CVF PDF'}})

PROGRESS.update({'gp_hsmm': {'level': 'M', 'note': 'evidence/M19_segmental_uncertainty.md', 'scope': '§3/Algorithm 2, selected §§4.1–4.2 setup, §5 limitations; chưa all tables/code', 'venue': 'Frontiers in Neurorobotics2017 — official DOI/full text'}, 'imprecise_semicrf': {'level': 'M', 'note': 'evidence/M19_segmental_uncertainty.md', 'scope': 'semi-CRF background, §§4.1–4.2, sleep setup, selected HNS/runtime; chưa supplementary/all experiments', 'venue': 'UAI2018 — official PMLR R16 reissued2026'}})

official_path = ROOT / 'official_metadata' / 'selected_records.json'
official = json.loads(official_path.read_text()) if official_path.exists() else []
records = []
for file in sorted((ROOT / 'sources').glob('*/acquisition.json')):
    acquisition = json.loads(file.read_text())
    key = acquisition['key']
    progress = PROGRESS.get(key, {'level':'unread','scope':'chỉ acquisition/metadata extraction, chưa có ghi chú đọc', 'venue':'chưa xác minh'})
    record = {**acquisition, 'reading_level':progress['level'], 'reading_scope':progress['scope'],
              'note':progress.get('note'), 'venue_status':progress['venue']}
    published = file.parent / 'published_acquisition.json'
    if published.exists():
        record['additional_published_artifact'] = json.loads(published.read_text())
    supplement = file.parent / 'supplement_acquisition.json'
    if supplement.exists():
        record['additional_supplementary_artifact'] = json.loads(supplement.read_text())
    matches = [x for x in official if any(acquisition['requested_id'] in z['url'] for z in x.get('links',[]))]
    if matches:
        record['official_venue_metadata'] = matches[0]
        record['venue_status'] = f"{matches[0]['venue']} — official CVF metadata acquired"
    records.append(record)
(ROOT / 'paper_registry.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
lines = ['# Reading ledger của lượt nghiên cứu mới', '',
 'Registry máy đọc: `third_party/proposal2_review_20261104/paper_registry.json`. Acquisition và reading là hai trạng thái khác nhau. Bảng này chỉ gán D khi đã có deep note do đọc thực tế; script tải file không tự gán D. Cờ C chỉ áp dụng các file và câu hỏi đã ghi, không hàm ý tái lập. Không có human-read attestation.', '',
 '| Paper/version | Venue verification | Mức đọc | Phạm vi/bằng chứng |', '|---|---|---|---|']
for record in records:
    version = ','.join(record.get('canonical_ids',[])) or record['requested_id']
    title = record.get('title') or record['key']
    scope = record['reading_scope']
    if record.get('note'):
        scope = f"[{scope}]({record['note']})"
    url = f"https://arxiv.org/abs/{version.split(',')[0]}" if record.get('canonical_ids') else record['artifacts'][0]['url']
    lines.append(f"| [{title}]({url}) `{version}` | {record['venue_status']} | {record['reading_level']} | {scope} |")
lines += ['', '## Cách đọc trạng thái', '', '- D là phạm vi đã đọc và có note, không phải mọi appendix/code đã được kiểm tra.', '- Các nguồn unread chỉ được acquired; không dùng làm evidence quyết định.', '- Motion/SLG syntheses và candidate comparison đã có. Main proposal là giả thuyết, chưa có training hoặc model reproduction.', '- Xem 10_completion_audit.md và RESEARCH_STATE.md để tra completion audit và giới hạn; số lượng nguồn không thay tiêu chí chất lượng.', '']
(DOC / '02_reading_ledger.md').write_text('\n'.join(lines))
print('Rendered',len(records),'acquisitions;',sum(x['reading_level'].startswith('D') for x in records),'with scoped deep notes.')
