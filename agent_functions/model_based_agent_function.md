# Model Based Agent Function

## Initialize Agent - `__init__(self)`
* Set agent name
* Set agent location to [0,0]
* Set agent direction to 'E'
* Set Pit probabilities (`P`) to `NUM_PITS` / 15
* Set Wumpus probabilities (`W`) to `NUM_WUMPI` / 15
* Set Gold probabilities (`G`) to `NUM_WUMPI` / 16
* Set `P[0][0] = 0`
* Set `W[0][0] = 0`

## Process - `process()`
* **Perceive** - `_update_state_pre_action(self, percept)`:
    * Update Pit probabilities based on percept - `_update_P(self, percept)`:
        * Compute `P(Pit|Breeze)` for all locations:
            * Compute Breeze probabilities (`B`)
            * Compute Pit probabilities if one pit was removed (`prior_P_removed`) (used when computing `P(B|P)`)
            * For each location, compute Pit probabilities (`P[x][y]`):
                * If there's a breeze:
                    * Compute `P(P_xy|B_percept_loc)` using Bayes Rule
                        * Compute prior Pit probs (`P(P)`)
                        * Compute Breeze probs (`P(B)`)
                        * Compute `P(P|B)`
                            * If prediction loc is adjacent to percept loc:
                                * return 1
                            * Otherwise:
                                * return Breeze prior probablity (given one pit is removed)
                        * Compute `P(P|B)` using Bayes Rule
                * If there's no breeze:
                    * Compute `P(P_xy|nB_percept_loc)` using Bayes Rule
                        * Compute prior Pit probs (`P(P)`)
                        * Compute Breeze probs (`P(B)`)
                        * compute `P(P|nB)`
                            * If prediction loc is adjacent to percept loc:
                                * return 0
                            * Otherwise:
                                * return Breeze prior probablity (given one pit is removed)
                        * Compute `P(P|nB)` using Bayes Rule

    * Update Wumpus probabilities based on percept - `_update_W(self, percept)`
    * Update Gold probabilities based on percept - `_update_G(self, percept)`
* Think - `_choose_action(self)`
* Act and update state - `_update_state_post_action(self, action)`
* Return `action`

