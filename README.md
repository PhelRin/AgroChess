♟️ Hyper-Aggressive Chess AI (AlphaZero-Style)

An AlphaZero-inspired Reinforcement Learning chess engine built from scratch in PyTorch. Unlike traditional engines that play objectively "perfect" chess, this bot is designed to learn how to play hyper-aggressive chess hence the name HyperChess

Released under the GNU General Public License (GPL).



✨ Features

1. Uses a ResNet backbone with separate Policy and Value heads, combined with a highly optimized Monte Carlo Tree Search (MCTS).
2. Automatically utilizes bfloat16, TF32 precision, and PyTorch DataLoaders to maximize GPU throughput on modern RTX cards.
3. The bot is mathematically penalized for long games, forcing it to calculate brutal, efficient checkmates.
4. The bot receives massive dopamine hits for capturing pieces within a 2-square radius of the enemy King, teaching it the art of the attacking sacrifice.
5. Evaluates draws as -0.8 (almost a total loss), making it refuse to trade Queens in equal endgames and prefer chaotic complications over peaceful draws.



🛠️ Prerequisites

You need a CUDA-enabled NVIDIA GPU to train this model in a reasonable amount of time.

Make sure you have Python 3.8+ installed, and install the required libraries:

    pip install torch numpy chess

(Note: Ensure your version of PyTorch is installed with CUDA support! Visit pytorch.org for the specific command for your system).

	
	            
🚀 Installation & Setup

Clone the repository:

    git clone https://github.com/PhelRin/HyperChess.git
    cd HyperChess  

(Optional) Add an Opening Book:
The code is designed to start games from dynamic, chaotic openings to force tactical play. Download a Polyglot .bin opening book (like ph-gambitbook.bin) and place it in the root folder.
If you don't use a book, the script will gracefully catch the error and just play from the standard starting chess position.



🧠 How to Train the Bot

To start generating self-play games and training the neural network, simply run:

    python main.py

What happens when you run this?

1. The script automatically creates a data_v2 folder.
2. The GPU will simulate thousands of games against itself.
3. Games that end in draws are thrown in the trash. Only decisive games (Checkmates/Resignations) are saved as .pkl files.

Note for Windows users: If the script crashes when transitioning from self-play to the training phase, open train.py and change num_workers=4 to num_workers=0 in the DataLoader.



⚔️ Play Against the Bot

Once the bot has generated a model.pt file, you can play against your creation directly in the terminal!

    python play.py

You will be prompted to choose White or Black. You input your moves using standard UCI format (e.g., e2e4, g1f3). The bot will use its trained Neural Network and MCTS to calculate its response.
