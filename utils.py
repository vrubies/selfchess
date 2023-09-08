import torch
import chess

def fen_to_808bits(fen):
    board_obj = chess.Board(fen)  # Create a chess.Board object
    turn = board_obj.turn
    castling = board_obj.castling_rights
    en_passant = board_obj.ep_square
    bitstring = ''
    
    # Encode board state
    for piece_type in chess.PIECE_TYPES:
        for color in [True, False]:  # True for White, False for Black
            # bitboard = chess.SquareSet(chess.BB_SQUARES)
            bitboard = board_obj.pieces(piece_type, color)
            bitstring += format(int(bitboard), '064b')  # 64-bit representation

    # Encode active color, 1 bit (1 for White, 0 for Black)
    bitstring += '1' if turn else '0'

    # Encode castling rights, 4 bits (KQkq)
    c_rights = [bool(castling & chess.BB_A1),
                bool(castling & chess.BB_H1),
                bool(castling & chess.BB_A8),
                bool(castling & chess.BB_H8)]

    bitstring += ''.join(['1' if state else '0' for state in c_rights])

    # Encode en passant possibility, 1 bit (1 if possible, 0 otherwise)
    bitstring += '1' if en_passant else '0'

    # Use remaining 34 bits for other game states (for demonstration, filling with zeros)
    bitstring += '0' * 34

    # Convert bitstring to PyTorch tensor of floats
    tensor = torch.tensor([float(bit) for bit in bitstring])

    return tensor