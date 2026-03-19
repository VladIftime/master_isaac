import os
import omni.kit.test
import omni.graph.core as og
import omni.graph.core.tests as ogts
from omni.graph.core.tests.omnigraph_test_utils import _TestGraphAndNode
from omni.graph.core.tests.omnigraph_test_utils import _test_clear_scene
from omni.graph.core.tests.omnigraph_test_utils import _test_setup_scene
from omni.graph.core.tests.omnigraph_test_utils import _test_verify_scene


class TestOgn(ogts.OmniGraphTestCase):

    async def test_data_access(self):
        test_file_name = "OgnOnTickTemplate.usda"
        usd_path = os.path.join(os.path.dirname(__file__), "usd", test_file_name)
        if not os.path.exists(usd_path):  # pragma: no cover
            self.assertTrue(False, f"{usd_path} not found for loading test")
        result, error = await ogts.load_test_file(usd_path)
        self.assertTrue(result, f"{error} on {usd_path}")
        test_node = og.Controller.node("/TestGraph/Template_omni_graph_action_OnTick")
        self.assertTrue(test_node.is_valid())
        node_type_name = test_node.get_type_name()
        self.assertEqual(og.GraphRegistry().get_node_type_version(node_type_name), 2)

        def _attr_error(
            attribute: og.Attribute, usd_test: bool
        ) -> str:  # pragma no cover
            test_type = "USD Load" if usd_test else "Database Access"
            return f"{node_type_name} {test_type} Test - {attribute.get_name()} value error"

        self.assertTrue(test_node.get_attribute_exists("inputs:framePeriod"))
        attribute = test_node.get_attribute("inputs:framePeriod")
        self.assertTrue(attribute.is_valid())
        expected_value = 0
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:onlyPlayback"))
        attribute = test_node.get_attribute("inputs:onlyPlayback")
        self.assertTrue(attribute.is_valid())
        expected_value = True
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("outputs:absoluteSimTime"))
        attribute = test_node.get_attribute("outputs:absoluteSimTime")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:deltaSeconds"))
        attribute = test_node.get_attribute("outputs:deltaSeconds")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:frame"))
        attribute = test_node.get_attribute("outputs:frame")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:isPlaying"))
        attribute = test_node.get_attribute("outputs:isPlaying")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:tick"))
        attribute = test_node.get_attribute("outputs:tick")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:time"))
        attribute = test_node.get_attribute("outputs:time")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:timeSinceStart"))
        attribute = test_node.get_attribute("outputs:timeSinceStart")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("state:accumulatedSeconds"))
        attribute = test_node.get_attribute("state:accumulatedSeconds")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("state:frameCount"))
        attribute = test_node.get_attribute("state:frameCount")
        self.assertTrue(attribute.is_valid())
