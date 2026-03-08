♟️ Hyper-Aggressive Chess AI (AlphaZero-Style)

An AlphaZero-inspired Reinforcement Learning chess engine built from scratch in PyTorch. Unlike traditional engines that play objectively "perfect" chess, this bot is mathematically hardcoded to be a hyper-aggressive, tactical assassin.

Through custom shaping rewards and a massive "Contempt Factor" for draws, this neural network learns to despise boring positional play. It aggressively fights for the center, sacrifices pieces to rip open the enemy King's pawn shield, and panics if the game goes past Move 30—forcing it to build lightning-fast mating nets.

Released under the GNU General Public License (GPL).

✨ Features

Custom AlphaZero Architecture: Uses a ResNet backbone with separate Policy and Value heads, combined with a highly optimized Monte Carlo Tree Search (MCTS).

Ampere/Blackwell Optimized: Automatically utilizes bfloat16, TF32 precision, and PyTorch DataLoaders to maximize GPU throughput on modern RTX cards.

The "Speedrun" Reward: The bot is mathematically penalized for long games, forcing it to calculate brutal, efficient checkmates.

The "Shield Buster" Reward: The bot receives massive dopamine hits for capturing pieces within a 2-square radius of the enemy King, teaching it the art of the attacking sacrifice.

Draw Contempt: Evaluates draws as -0.8 (almost a total loss), making it refuse to trade Queens in equal endgames and prefer chaotic complications over peaceful draws.

🛠️ Prerequisites

You need a CUDA-enabled NVIDIA GPU to train this model in a reasonable amount of time.

Make sure you have Python 3.8+ installed, and install the required libraries:

pip install torch numpy chess

(Note: Ensure your version of PyTorch is installed with CUDA support! Visit pytorch.org for the specific command for your system).

🚀 Installation & Setup

Clone the repository:

git clone https://github.com/YOUR-USERNAME/aggressive-chess-ai.git
cd aggressive-chess-ai

(Optional) Add an Opening Book:
The code is designed to start games from dynamic, chaotic openings to force tactical play. Download a Polyglot .bin opening book (like ph-gambitbook.bin) and place it in the root folder.
If you don't use a book, the script will gracefully catch the error and just play from the standard starting chess position.

🧠 How to Train the Bot

To start generating self-play games and training the neural network, simply run:

python main.py

What happens when you run this?

The script automatically creates a data_v2 folder.

The GPU will simulate thousands of games against itself.

Games that end in draws are thrown in the trash. Only decisive games (Checkmates/Resignations) are saved as .pkl files.

A sliding-window Replay Buffer grabs the most recent games, trains the ResNet, and overwrites model.pt with a slightly smarter brain.

Note for Windows users: If the script crashes when transitioning from self-play to the training phase, open train.py and change num_workers=4 to num_workers=0 in the DataLoader.

⚔️ Play Against the Bot

Once the bot has generated a model.pt file, you can play against your creation directly in the terminal!

python play.py

You will be prompted to choose White or Black. You input your moves using standard UCI format (e.g., e2e4, g1f3). The bot will use its trained Neural Network and MCTS to calculate its response. Good luck surviving the middlegame!

📂 File Structure

main.py: The orchestrator that loops self-play and training.

selfplay.py: Generates MCTS games and applies the custom aggressive shaping rewards.

train.py: Loads the .pkl files and executes backpropagation on the neural network.

model.py: The PyTorch ResNet architecture (Policy and Value heads).

mcts.py: The Monte Carlo Tree Search algorithm with Dirichlet noise and virtual loss batching.

env.py: The custom chess environment and AlphaZero action encoder (4672 possible move planes).

play.py: The interactive human-vs-bot terminal script.
