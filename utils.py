import torch
import chess
import numpy as np
import time
from numba import jit

def fen_to_halfkp(fen, binary=False):
    board = chess.Board(fen)
    turn = board.turn # 2 options
    # castling = board.castling_rights # 2^4 = 16
    castling_white = np.array([
        board.has_kingside_castling_rights(1),
        board.has_queenside_castling_rights(1),
        board.has_kingside_castling_rights(0),
        board.has_queenside_castling_rights(0)
    ])
    castling_black = np.array([
        board.has_kingside_castling_rights(0),
        board.has_queenside_castling_rights(0),
        board.has_kingside_castling_rights(1),
        board.has_queenside_castling_rights(1)
    ])
    # compress_castling_rights_white(castling)
    # castling_black = compress_castling_rights_black(castling)
    en_passant = board.ep_square # 16 * 2 = 32 options
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)
    # Create array of pieces
    piece_types = np.zeros(64, dtype=np.int32)  # 0 represents no piece
    piece_colors = np.zeros(64, dtype=np.int32)  # 0 for white, 1 for black, -1 for no piece
    mirror = np.zeros(64, dtype=np.int32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        mirror[square] = chess.square_mirror(square)
        if piece:
            piece_types[square] = piece.piece_type
            piece_colors[square] = 0 if piece.color == chess.WHITE else 1
        else:
            piece_types[square] = 0
            piece_colors[square] = -1

    # return fen_to_halfkp_jit(white_king_sq, black_king_sq, piece_types, piece_colors, mirror, turn, castling, en_passant, binary=binary)
    # try:
    bitstring_ego, bitstring_alter = fen_to_halfkp_jit(turn, white_king_sq, black_king_sq, piece_types, piece_colors, mirror, castling_white, castling_black, en_passant)
    # except Exception as e:
    #     import pdb
    #     pdb.set_trace()


    if binary:
        # # Ensure the bitstring length is a multiple of 8
        # if len(bitstring_ego) % 8 != 0:
        #     padding = '0' * (8 - len(bitstring_ego) % 8)
        #     bitstring_ego += padding
        #     bitstring_alter += padding
        # Convert bitstring to bytes outside of the JIT function
        byte_representation_ego = bitstring_ego.tobytes()#int(bitstring_ego, 2).to_bytes(len(bitstring_ego) // 8, byteorder='big')
        byte_representation_alter = bitstring_alter.tobytes()#int(bitstring_alter, 2).to_bytes(len(bitstring_alter) // 8, byteorder='big')
        # byte_representation_ego = bytes(bitstring_ego) #[bitstring_ego[i:i+8] for i in range(0, len(bitstring_ego), 8)])
        # byte_representation_alter = bytes(bitstring_alter) #[bitstring_alter[i:i+8] for i in range(0, len(bitstring_alter), 8)])

        return byte_representation_ego, byte_representation_alter

    return bitstring_ego, bitstring_alter

@jit(nopython=True)
def fen_to_halfkp_jit(turn, white_king_sq, black_king_sq, piece_types, piece_colors, mirror, castling_white, castling_black, en_passant):
    bitstring_white = np.zeros(45056 + 4 + 16 + 4, dtype=np.bool_)
    bitstring_black = np.zeros(45056 + 4 + 16 + 4, dtype=np.bool_)

    # Function to calculate index in the feature array
    def halfkp_index(ego_king, piece_sq, piece_type, color):

        # Compute indexes assuming the piece_type is not the opponent's king
        if (piece_type != chess.KING):
          piece_index = (piece_type - 1) * 64 + piece_sq  # 64 squares for each piece type (10 - 1)
          color_index = (piece_index * 2) + color
        else:
          # the last 64 indexes are reserved for the non-ego king
          color_index = (5 * 64 * 2) + piece_sq
        
        index = (color_index * 64) + ego_king     
        return index

    # Initialize the feature vector
    # 5 pieces, of 2 colors, in 64 spots + 64 spots alter king and 64 spots for ego king = 40960 
    halfkp_features_white = np.zeros(45056, dtype=np.bool_)  # The size of the HalfKP feature set
    halfkp_features_black = np.zeros(45056, dtype=np.bool_)

    enpassant_features_white = np.zeros(16, dtype=np.bool_)  # The size of the HalfKP feature set
    enpassant_features_black = np.zeros(16, dtype=np.bool_)

    # ========== WHITE ============

    # For white king
    for square, piece_elem in enumerate(zip(piece_types, piece_colors)):
        if piece_elem[0]:
            piece_type, color = piece_elem
            # if king is black and black turn ignore, if king is white and white turn ignore
            if(piece_type == chess.KING and color == chess.WHITE):
                continue
            
            idx = halfkp_index(white_king_sq, square, piece_type, color)
            if idx is not None:
                halfkp_features_white[idx] = 1

    # Convert the feature vector to bitstring
    # bitstring_white = ''.join(['1' if bit else '0' for bit in halfkp_features_white])
    bitstring_white[:45056] = halfkp_features_white[:]

    # Encode castling rights, 4 bits (KQkq)
    # c_rights = np.array([bool(castling & chess.BB_H1),
    #                     bool(castling & chess.BB_A1),
    #                     bool(castling & chess.BB_H8),
    #                     bool(castling & chess.BB_A8)])

    # bitstring_white += ''.join(['1' if state else '0' for state in castling_white])
    bitstring_white[45056:(45056+4)] = castling_white[:]

    # try:
    # Encode en passant square
    if en_passant is not None:
        if 16 <= en_passant <= 23:  # 3rd rank (for White pawns)
            en_passant_index = en_passant - 16
            enpassant_features_white[en_passant_index] = 1
        elif 40 <= en_passant <= 47:  # 6th rank (for Black pawns)
            en_passant_index = en_passant - 40
            enpassant_features_white[en_passant_index + 8] = 1

    # print(enpassant_features_white)
    # bitstring_white += ''.join(['1' if bit else '0' for bit in enpassant_features_white])
    bitstring_white[(45056+4):(45056+4+16)] = enpassant_features_white[:]
    
    # except Exception as e:
    #     print("EPIC FAIL 1")

    # ========== BLACK ============

    # For black king
    for square, piece_elem in enumerate(zip(piece_types, piece_colors)):
        if piece_elem[0]:
            piece_type, color = piece_elem
            # if king is black and black turn ignore, if king is white and white turn ignore
            if(piece_type == chess.KING and color == chess.BLACK):
                continue
            
            idx = halfkp_index(mirror[black_king_sq], mirror[square], piece_type, not color)
            if idx is not None:
                halfkp_features_black[idx] = 1

    # # Convert the feature vector to bitstring
    # bitstring_black = ''.join(['1' if bit else '0' for bit in halfkp_features_black])
    bitstring_black[:45056] = halfkp_features_black[:]

    # Encode castling rights, 4 bits (KQkq)
    # c_rights = np.array([bool(castling & chess.BB_H8),
    #                     bool(castling & chess.BB_A8),
    #                     bool(castling & chess.BB_A1),
    #                     bool(castling & chess.BB_H1)])

    bitstring_black[45056:(45056+4)] = castling_black[:]
    # bitstring_black += ''.join(['1' if state else '0' for state in castling_black])

    # Encode en passant square
    # try:
    if en_passant is not None:
        if 16 <= mirror[en_passant] <= 23:  # 3rd rank (for White pawns)
            en_passant_index = mirror[en_passant] - 16
            enpassant_features_black[en_passant_index] = 1
        elif 40 <= mirror[en_passant] <= 47:  # 6th rank (for Black pawns)
            en_passant_index = mirror[en_passant] - 40
            enpassant_features_black[en_passant_index + 8] = 1 
    # print(enpassant_features_black)
    # bitstring_black += ''.join(['1' if bit else '0' for bit in enpassant_features_black])    
    bitstring_black[(45056+4):(45056+4+16)] = enpassant_features_black[:]
    # except Exception as e:
    #     print("EPIC FAIL 2")
    
    if (turn == chess.WHITE):
      return bitstring_white, bitstring_black
    else:
      return bitstring_black, bitstring_white 
    
def tic():
    """Start a timer."""
    global start_time
    start_time = time.time()

def toc():
    """Stop the timer and print the elapsed time."""
    elapsed_time = time.time() - start_time
    return elapsed_time
    # print(f"Elapsed time: {elapsed_time} seconds")

# # Example usage
# fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"  # Starting position

# sum_total = 0
# iters = 5000
# for i in range(iters):
#   tic()
#   [ego_bin, alter_bin] = fen_to_halfkp(fen, binary=True)
#   sum_total += toc()
# print(f"Elapsed time: {sum_total / iters} seconds")

# [ego_bin, alter_bin] = fen_to_halfkp(fen, binary=True)
# [ego_bit, alter_bit] = fen_to_halfkp(fen, binary=False)

# print("Binary (Bytes):", len(ego_bin), len(alter_bin))
# print("Bitstring:", len(ego_bit), len(alter_bit))