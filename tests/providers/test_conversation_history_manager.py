"""
Test script for OM1 Conversation Persistence Feature
Tests the conversation history save/load functionality
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
import shutil
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from providers.conversation_history_manager import ConversationHistoryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def add_result(self, test_name: str, passed: bool, message: str = ""):
        self.tests.append({
            "name": test_name,
            "passed": passed,
            "message": message
        })
        if passed:
            self.passed += 1
            logger.info(f"✓ {test_name}: PASSED")
        else:
            self.failed += 1
            logger.error(f"✗ {test_name}: FAILED - {message}")
    
    def print_summary(self):
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total Tests: {self.passed + self.failed}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {(self.passed/(self.passed+self.failed)*100):.1f}%")
        print("="*70)
        
        if self.failed > 0:
            print("\nFailed Tests:")
            for test in self.tests:
                if not test["passed"]:
                    print(f"  - {test['name']}: {test['message']}")


def test_basic_initialization():
    """Test 1: Basic initialization"""
    try:
        test_dir = "test_conversations"
        manager = ConversationHistoryManager(
            history_dir=test_dir,
            agent_name="test_agent",
            auto_save=False
        )
        
        # Verify directory created
        assert Path(test_dir).exists(), "History directory not created"
        
        # Verify empty history
        assert len(manager.history) == 0, "History should be empty on first init"
        
        # Cleanup
        shutil.rmtree(test_dir)
        
        return True, ""
    except Exception as e:
        return False, str(e)


def test_add_and_retrieve_messages():
    """Test 2: Add and retrieve messages"""
    try:
        test_dir = "test_conversations"
        manager = ConversationHistoryManager(
            history_dir=test_dir,
            agent_name="test_agent",
            auto_save=False
        )
        
        # Add messages
        manager.add_message("system", "You are a helpful assistant")
        manager.add_message("user", "Hello!")
        manager.add_message("assistant", "Hi there!")
        
        # Verify count
        assert len(manager.history) == 3, f"Expected 3 messages, got {len(manager.history)}"
        
        # Verify content
        history = manager.get_history()
        assert history[0]["role"] == "system"
        assert history[1]["content"] == "Hello!"
        assert history[2]["role"] == "assistant"
        
        # Cleanup
        shutil.rmtree(test_dir)
        
        return True, ""
    except Exception as e:
        return False, str(e)


def test_persistence():
    """Test 3: Save and load persistence"""
    try:
        test_dir = "test_conversations"
        agent_name = "test_persistence_agent"
        
        # First session: Create and save
        manager1 = ConversationHistoryManager(
            history_dir=test_dir,
            agent_name=agent_name,
            auto_save=True
        )
        
        manager1.add_message("user", "Remember this message")
        manager1.add_message("assistant", "I will remember it")
        
        # Verify file exists
        history_file = Path(test_dir) / f"{agent_name}_history.json"
        assert history_file.exists(), "History file not created"
        
        # Read file directly
        with open(history_file, 'r') as f:
            data = json.load(f)
        assert data["message_count"] == 2, "Wrong message count in file"
        
        # Second session: Load existing
        manager2 = ConversationHistoryManager(
            history_dir=test_dir,
            agent_name=agent_name,
            auto_save=False
        )
        
        # Verify loaded correctly
        assert len(manager2.history) == 2, f"Expected 2 messages, got {len(manager2.history)}"
        assert manager2.history[0]["content"] == "Remember this message"
        assert manager2.history[1]["content"] == "I will remember it"
        
        # Cleanup
        shutil.rmtree(test_dir)
        
        return True, ""
    except Exception as e:
        return False, str(e)


def test_history_limit():
    """Test 4: History size limit enforcement"""
    try:
        test_dir = "test_conversations"
        max_size = 5
        
        manager = ConversationHistoryManager(
            history_dir=test_dir,
            agent_name="test_limit_agent",
            auto_save=False,
            max_history_size=max_size
        )
        
        # Add more messages than limit
        for i in range(10):
            manager.add_message("user", f"Message {i}")
        
        # Verify limit enforced
        assert len(manager.history) == max_size, f"Expected {max_size} messages, got {len(manager.history)}"
        
        # Verify kept the most recent
        assert manager.history[-1]["content"] == "Message 9"
        assert manager.history[0]["content"] == "Message 5"
        
        # Cleanup
        shutil.rmtree(test_dir)
        
        return True, ""
    except Exception as e:
        return False, str(e)


def test_formatted_history():
    """Test 5: Formatted history for LLM"""
    try:
        test_dir = "test_conversations"
        manager = ConversationHistoryManager(
            history_dir=test_dir,
            agent_name="test_format_agent",
            auto_save=False
        )
        
        # Add various message types
        manager.add_message("system", "System prompt")
        manager.add_message("user", "User question")
        manager.add_message("assistant", "Assistant response")
        
        # Get formatted (with system)
        formatted_with_system = manager.get_formatted_history(include_system=True)
        assert len(formatted_with_system) == 3
        assert "timestamp" not in formatted_with_system[0]  # Should be stripped
        
        # Get formatted (without system)
        formatted_without_system = manager.get_formatted_history(include_system=False)
        assert len(formatted_without_system) == 2
        assert formatted_without_system[0]["role"] == "user"
        
        # Cleanup
        shutil.rmtree(test_dir)
        
        return True, ""
    except Exception as e:
        return False, str(e)


def test_export_import():
    """Test 6: Export and import functionality"""
    try:
        test_dir = "test_conversations"
        export_file = "test_export.json"
        
        # Create manager with data
        manager1 = ConversationHistoryManager(
            history_dir=test_dir,
            agent_name="test_export_agent",
            auto_save=False
        )
        
        manager1.add_message("user", "Export test message 1")
        manager1.add_message("assistant", "Export test message 2")
        
        # Export
        manager1.export_history(export_file)
        assert Path(export_file).exists(), "Export file not created"
        
        # Create new manager and import
        manager2 = ConversationHistoryManager(
            history_dir=test_dir,
            agent_name="test_import_agent",
            auto_save=False
        )
        
        manager2.import_history(export_file)
        assert len(manager2.history) == 2, "Import failed"
        assert manager2.history[0]["content"] == "Export test message 1"
        
        # Cleanup
        shutil.rmtree(test_dir)
        Path(export_file).unlink()
        
        return True, ""
    except Exception as e:
        return False, str(e)


def test_clear_with_backup():
    """Test 7: Clear history with backup"""
    try:
        test_dir = "test_conversations"
        manager = ConversationHistoryManager(
            history_dir=test_dir,
            agent_name="test_clear_agent",
            auto_save=False
        )
        
        # Add messages
        manager.add_message("user", "Message to backup")
        manager.add_message("assistant", "Response to backup")
        
        # Clear with backup
        manager.clear_history(save_backup=True)
        
        # Verify cleared
        assert len(manager.history) == 0, "History not cleared"
        
        # Verify backup exists
        backup_files = list(Path(test_dir).glob("test_clear_agent_backup_*.json"))
        assert len(backup_files) > 0, "Backup file not created"
        
        # Cleanup
        shutil.rmtree(test_dir)
        
        return True, ""
    except Exception as e:
        return False, str(e)


def test_concurrent_access():
    """Test 8: Thread-safe concurrent access"""
    try:
        import threading
        
        test_dir = "test_conversations"
        manager = ConversationHistoryManager(
            history_dir=test_dir,
            agent_name="test_concurrent_agent",
            auto_save=False
        )
        
        # Function to add messages concurrently
        def add_messages(thread_id, count):
            for i in range(count):
                manager.add_message("user", f"Thread {thread_id} message {i}")
        
        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=add_messages, args=(i, 10))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Verify all messages added
        assert len(manager.history) == 50, f"Expected 50 messages, got {len(manager.history)}"
        
        # Cleanup
        shutil.rmtree(test_dir)
        
        return True, ""
    except Exception as e:
        return False, str(e)


def run_all_tests():
    """Run all tests and report results"""
    results = TestResults()
    
    print("\n" + "="*70)
    print("OM1 CONVERSATION PERSISTENCE TEST SUITE")
    print("="*70 + "\n")
    
    # Run tests
    tests = [
        ("Basic Initialization", test_basic_initialization),
        ("Add and Retrieve Messages", test_add_and_retrieve_messages),
        ("Persistence (Save/Load)", test_persistence),
        ("History Size Limit", test_history_limit),
        ("Formatted History", test_formatted_history),
        ("Export/Import", test_export_import),
        ("Clear with Backup", test_clear_with_backup),
        ("Concurrent Access", test_concurrent_access),
    ]
    
    for test_name, test_func in tests:
        try:
            passed, message = test_func()
            results.add_result(test_name, passed, message)
        except Exception as e:
            results.add_result(test_name, False, f"Unexpected error: {e}")
    
    # Print summary
    results.print_summary()
    
    return results.failed == 0


if __name__ == "__main__":
    print("\nStarting OM1 Conversation Persistence Tests...\n")
    
    success = run_all_tests()
    
    if success:
        print("\n✓ ALL TESTS PASSED! Conversation persistence is working correctly.")
        print("✓ Ready for integration into OM1")
        sys.exit(0)
    else:
        print("\n✗ SOME TESTS FAILED. Please review the errors above.")
        sys.exit(1)