from typing import TYPE_CHECKING, Any, Dict
from manager import Manager
from runner import Runner
from locales import LOCALES
from chatter import Chatter

if TYPE_CHECKING:
    from gradio.components import Component


class Engine:
    def __init__(self) -> None:
        self.manager = Manager()
        self.runner = Runner(self.manager)
        self.chatter = Chatter(runner = self.runner, manager = self.manager)
        
    def _update_component(
        self,
        input_dict: Dict[str, Dict[str, Any]]
    ) -> Dict["Component", "Component"]:
        output_dict: Dict["Component", "Component"] = {}
        for elem_id, elem_attr in input_dict.items():
            elem = self.manager.get_elem_by_id(elem_id)
            output_dict[elem] = elem.__class__(**elem_attr)
        return output_dict
    
    def change_lang(self, lang: str):
        return {
            elem: elem.__class__(**LOCALES[elem_name][lang])
            for elem_name, elem in self.manager.get_elem_iter()
            if elem_name in LOCALES
        }
        
    def resume(self):
        base_lang = 'en'
        init_dict = {'basic.lang': {'value': base_lang}}
        yield self._update_component(init_dict)
