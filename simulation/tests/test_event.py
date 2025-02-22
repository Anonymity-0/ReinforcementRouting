import unittest
from ..event import Event, PacketEvent, LinkEvent, TopologyEvent, EventType

class TestEvent(unittest.TestCase):
    """测试事件系统"""
    
    def test_event_creation(self):
        """测试事件创建"""
        event = Event(
            time=1.0,
            type="test",
            data={"test": "data"},
            priority=1
        )
        
        self.assertEqual(event.time, 1.0)
        self.assertEqual(event.type, "test")
        self.assertEqual(event.data["test"], "data")
        self.assertEqual(event.priority, 1)
    
    def test_packet_event(self):
        """测试数据包事件"""
        event = PacketEvent(
            time=1.0,
            event_type=EventType.PACKET_ARRIVAL,
            packet_id="test_packet",
            src=1,
            dst=2,
            current_node=1,
            size=1000,
            qos_class=0
        )
        
        self.assertEqual(event.time, 1.0)
        self.assertEqual(event.type, EventType.PACKET_ARRIVAL)
        self.assertEqual(event.data["packet_id"], "test_packet")
        self.assertEqual(event.data["src"], 1)
        self.assertEqual(event.data["dst"], 2)
        self.assertEqual(event.data["current_node"], 1)
        self.assertEqual(event.data["size"], 1000)
        self.assertEqual(event.data["qos_class"], 0)
        self.assertEqual(event.data["creation_time"], 1.0)
    
    def test_link_event(self):
        """测试链路事件"""
        event = LinkEvent(
            time=1.0,
            event_type=EventType.LINK_UPDATE,
            src=1,
            dst=2,
            capacity=100.0,
            delay=0.01
        )
        
        self.assertEqual(event.time, 1.0)
        self.assertEqual(event.type, EventType.LINK_UPDATE)
        self.assertEqual(event.data["src"], 1)
        self.assertEqual(event.data["dst"], 2)
        self.assertEqual(event.data["capacity"], 100.0)
        self.assertEqual(event.data["delay"], 0.01)
    
    def test_topology_event(self):
        """测试拓扑事件"""
        event = TopologyEvent(
            time=1.0,
            event_type=EventType.TOPOLOGY_UPDATE,
            node_id=1,
            position=(0.0, 0.0, 0.0),
            velocity=(1.0, 1.0, 1.0)
        )
        
        self.assertEqual(event.time, 1.0)
        self.assertEqual(event.type, EventType.TOPOLOGY_UPDATE)
        self.assertEqual(event.data["node_id"], 1)
        self.assertEqual(event.data["position"], (0.0, 0.0, 0.0))
        self.assertEqual(event.data["velocity"], (1.0, 1.0, 1.0))

if __name__ == '__main__':
    unittest.main() 