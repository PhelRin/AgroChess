# AggroChess v1.0.0 ⚔️

Welcome to the initial release of **AggroChess**, a tactical, highly aggressive chess engine written in Rust. Unlike traditional engines that play dry, slow-grinding positional chess, AggroChess is specifically tuned to mimic the active, sacrifice-heavy playstyle of World Champion Mikhail Tal. 

This release contains the precompiled standalone executable binary `aggro_chess.exe`.

### Key Features in this Release:
*   **Mikhail Tal-Style Aggression Heuristics**: Includes speculative sacrifice discounts (reducing material deficit calculations by up to 45% during attacks) and king ring virtual mobility escape restrictions (+150 cp cramp bonuses).
*   **Attack Coordination Heuristic**: Awards a positional bonus of **+25 cp** for each square in the opponent's king ring targeted by $\ge 2$ coordinating friendly pieces.
*   **Built-in Opening Gambit Book**: Hardcoded prefix-path-matching for sharp openings, including the *Evans Gambit, King's Gambit, Smith-Morra Gambit, Danish Gambit, Albin Countergambit, Stafford Gambit,* and *Budapest Gambit* (operating via Zobrist hashes even if no external opening book is loaded).
*   **King Safety Balance**: Center-king open-file penalties and castling-wing bonuses encourage early castling under quiet positions, while allowing the king to lead the assault from the center in dominant attacks.
*   **Optimized Search Engine**: Principal Variation Search (PVS), Aspiration Windows, killer/history move ordering, and Static Exchange Evaluation (SEE) capture pruning. Runs at **1.4+ Million Nodes Per Second (NPS)**.

---

### Installation & Setup Instructions

Since this release only includes the precompiled standalone executable, follow these steps to play against it:

1.  **Download the Binary**: Download the `aggro_chess.exe` file attached to this release.
2.  **Load in a Chess GUI**:
    *   **Arena Chess GUI**: Go to `Engines` -> `New Engine` -> Select `UCI` -> Browse and select the downloaded `aggro_chess.exe`.
    *   **Cute Chess / Lichess / ChessBase**: Add the engine as a standard UCI engine pointing to the path of `aggro_chess.exe`.
3.  **Opening Book Setup (Optional)**:
    *   By default, the engine searches for `ph-gambitbook.bin` in the folder it is executed from to supplement non-gambit openings.
    *   To play **strictly the built-in curated gambits**, configure the engine option `BookPath` to a non-existent file (e.g. `none.bin`).
4.  **TT Hash Customization**: Adjust the `Hash` option in your GUI to change search memory size in Megabytes (defaults to 16MB).
