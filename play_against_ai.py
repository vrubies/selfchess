from IPython.display import display, clear_output
from random import randrange
import chess
import chess.svg
import torch
# from evaluation_dataset import *
from train_regressive_model import EvaluationModel, config
from utils import fen_to_808bits

# Initialize the model first
model = EvaluationModel(layer_count=config["layer_count"],batch_size=config["batch_size"],learning_rate=1e-3)

# Load the model state dictionary
model.load_state_dict(torch.load("1694085972-batch_size-512-layer_count-4model.pth"))

# If you're planning to use the model only for inference, it's good to set it to evaluation mode
model.eval()

board = chess.Board() #"8/7P/k7/8/8/8/8/K7 w - - 0 1")
display(chess.svg.board(board, size=350))

while not board.is_game_over():
    if board.turn == chess.WHITE:
        # Your move logic here
        user_input_valid = False
        while not user_input_valid:
            user_move = input("Enter your move (e.g., 'e2e4'): ")
            if len(user_move) == 4 or len(user_move) == 5:
                try:
                    move = chess.Move.from_uci(user_move)
                    if move in board.legal_moves:
                        user_input_valid = True
                    else:
                        print("Illegal move. Try again.")
                except:
                    print("Invalid input. Try again.")
            else:
                print("Input must be 4 characters long. Try again.")
    else:
        # AI logic
        max_value = -float('inf')  # Initialize to positive infinity
        best_move = None

        for move in board.legal_moves:
            board.push(move)
            
            # Convert board state to FEN, then to 808-bit tensor
            fen = board.fen()
            input_tensor = fen_to_808bits(fen)

            # Evaluate board state using neural network
            with torch.no_grad():
                output = model(input_tensor)  # Assuming model expects a batch dimension

            # Update the best move if this move has a lower evaluation value
            if output.item() > max_value:
                max_value = output.item()
                best_move = move

            board.pop()  # Undo the move to return to the original state

        move = best_move

    # Execute move and update board
    board.push(move)
    clear_output(wait=True)  # Clear the old board
    display(chess.svg.board(board, size=350))

# After the loop, you can check the game's outcome
if board.is_checkmate():
    print("Checkmate")
elif board.is_stalemate():
    print("Stalemate")
elif board.is_insufficient_material():
    print("Draw due to insufficient material")
elif board.is_seventyfive_moves():
    print("Draw due to 75-move rule")
elif board.is_fivefold_repetition():
    print("Draw due to fivefold repetition")
elif board.is_variant_draw():
    print("Draw due to variant-specific rules")
else:
    print("Draw due to fifty-move rule or threefold repetition")