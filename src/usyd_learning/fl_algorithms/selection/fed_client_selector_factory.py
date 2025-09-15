from __future__ import annotations

from .fed_client_selector_args import FedClientSelectorArgs
from .fed_client_selector_abc import FedClientSelector


class FedClientSelectorFactory:
    '''
    Fed client selector factory
    '''

    @staticmethod
    def create_args(config_dict: dict, is_clone_dict: bool = False) -> FedClientSelectorArgs:
        """
        Static method to create client selector args
        """
        return FedClientSelectorArgs(config_dict, is_clone_dict)
    
    @staticmethod
    def create(args: FedClientSelectorArgs) -> FedClientSelector:        
        """
        Static method to create client selector
        """
        match args.select_method:
            case "all":
                from .methods._fed_client_selector_all import FedClientSelector_All
                selector = FedClientSelector_All(args)
            case "high_loss":
                from .methods._fed_client_selector_high_loss import FedClientSelector_HighLoss
                selector = FedClientSelector_HighLoss(args)
            case "random":
                from .methods._fed_client_selector_random import FedClientSelector_Random
                selector = FedClientSelector_Random(args)
            case _:
                raise Exception(f"Fed client selector method '{args.select_method}' not found")        

        return selector
