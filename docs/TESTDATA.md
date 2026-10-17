# Test data

Nothing here is committed: videos are ignored by .gitignore and stay on your machine. This file only logs where clips come from.

## Your own footage (read in place, never copied, never uploaded, never looked at by a model)

- One phone video of your own and the target clips cut from its detect run. Their paths are set per machine (`inframe.localcfg`: `INFRAME_REGRESS_VIDEO`,
  `INFRAME_REGRESS_BASELINE`, `INFRAME_SPLICE_TARGET_DIR`), never in the repo (the regression baseline, `tools/regress.py`). Without them those gates print SKIP. The folder of target clips also holds `target_windows.json` (for each of `targetA`, `targetB`, `targetC`: the file's duration and the first and last second the target is in frame), which `tools/splice.py` reads.
- The spliced composites use three of those clips (`targetA`, `targetB`, `targetC` in `tools/splice.py`, found by their index `*_01_*`, `*_03_*`, `*_05_*`). The composites are written to
  `~/.cache/inframe/splice/` (outside the repo) together with their truth JSON.

## Openly licensed clips of other people (used as "other" tiles)

Local copies live in a folder set per machine (`INFRAME_DEV_SAMPLES`, see `inframe.localcfg`; not in the repo); each has a `.source.txt` with page, author and license.

| key in splice.py | file | what is in it | author | license |
|---|---|---|---|---|
| pair1 | jump/cmj_squatjump_front_2people_pexels_8857689_danilyuk.mp4 | two people exercising | Pavel Danilyuk, Pexels | Pexels License |
| pair2 | jump/cmj_squatjump_front_2people_pexels_8858442_danilyuk.mp4 | two people exercising | Pavel Danilyuk, Pexels | Pexels License |
| bystanders | skateboard/kickflip/hippyjump_bystanders_slowmo_commons.webm | skater with bystanders | BoarderleinTV, Wikimedia Commons | CC BY 3.0 |
| lone1 | skateboard/ollie_front_pexels_2791955_cottonbro.mp4 | one skater, vertical | cottonbro studio, Pexels | Pexels License |
| lone2 | skateboard/kickflip/kickflip_front_pexels_4824363_podrez.mp4 | one skater | Artem Podrez, Pexels | Pexels License |

Not used: the Mixkit clips in that folder (personal, non-commercial use only).

## Fetched for T1b (Pexels or Wikimedia Commons only, at most 200 MB in all)

Downloaded with curl from Wikimedia Commons (upload.wikimedia.org) into `~/.cache/inframe/testdata/` (outside the repo, 88 MB in all). Pexels was tried first and refused scripted access (HTTP 403 for page and file URLs), so nothing was taken from it.
The camera motion of the pan clips was measured, not looked at: median horizontal shift between samples 5 fps apart is 16 px of 320 for pan1 (all one direction) and 3.5 px for pan2, under 0.2 px for the static ones. The person counts come from the detector.

| key in splice.py | file | what is in it | author | license | URL |
|---|---|---|---|---|---|
| dogs1 | Goldendoodle and Black Lab puppies playing in slow motion.webm | two dogs playing (slow motion, 64 s) | Tim Sweet | CC BY-SA 3.0 | https://commons.wikimedia.org/wiki/File:Goldendoodle_and_Black_Lab_puppies_playing_in_slow_motion.webm |
| dogs2 | Two poodles playing.ogv | two dogs playing (31 s) | Living in Monrovia (A W) | CC BY-SA 2.0 | https://commons.wikimedia.org/wiki/File:Two_poodles_playing.ogv |
| cars1 | Bangkok Traffic During Enduring Partners 26 (1009429).webm | street traffic with cars and a few people (24 s, AV1) | U.S. Army National Guard video by Staff Sgt. James Bunn | Public domain | https://commons.wikimedia.org/wiki/File:Bangkok_Traffic_During_Enduring_Partners_26_(1009429).webm |
| crowd1 | Calle preciados.ogv | busy street, about 9 people in shot, static camera (19 s) | OsvaldoGago www.fotografar.net | CC BY-SA 3.0 | https://commons.wikimedia.org/wiki/File:Calle_preciados.ogv |
| pan1 | Dog walking in the Mission in San Francisco.ogv | camera follows a person walking a dog: a long pan (17 s) | Sage Ross | CC BY-SA 3.0 | https://commons.wikimedia.org/wiki/File:Dog_walking_in_the_Mission_in_San_Francisco.ogv |
| pan2 | Central Park Connaught Place VID 20210411 175602.webm | handheld pan over a crowd (5 s) | Sumita Roy Dutta | CC BY-SA 4.0 | https://commons.wikimedia.org/wiki/File:Central_Park_Connaught_Place_VID_20210411_175602.webm |

CC BY and CC BY-SA need attribution (author and license above). The U.S. Army clip is public domain. The composites are never published, and no clip is committed.

Still not sourced: a car clip with the car as the main subject (cars1 is a street with cars, a few people in it). A 1 s CC0 car clip (File:Coche con placas de Sinaloa, 4 de enero de 2026.webm) was downloaded and deleted: too short to use.

## Composites (`tools/splice.py build`)

Ten composites, 21 to 46 s each, 1280x720 at 30 fps, no audio. The truth JSON lists, for every shot, the tile rectangles and
which tile holds the target; `tools/splice.py score TRUTH SEGMENTS` turns a run's kept seconds into time-weighted precision and recall.

| name | what it tests |
|---|---|
| solo_gaps | the target alone, with other lone people in between |
| two_side | the target beside two people, sides change between shots |
| swap_sides | the target and another person trade places left/right between shots |
| quad | the target in a 2x2 grid with three others, then without the target |
| pip_small | the target as a small picture-in-picture over a clip with bystanders |
| dogs_cars | the target beside dogs, beside traffic, and in a 2x2 grid with dogs and traffic: dogs and cars must not become the target |
| crowd_quad | the target in a 2x2 grid with a busy street (9 people), a panning shot, two people |
| pan_pip | the target in a large picture-in-picture over the panning clips |
| pan_follow | no target: the two panning clips alone, only for the BoT-SORT ID-switch count (role `tracked`) |
| outfit_changed | the target unchanged, then the same footage with hue +140 degrees (a different outfit colour than the first excerpt, for photo mode in T1c) |
