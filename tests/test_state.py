#type: ignore
from NovaHarness.harness.messages import Message
from NovaHarness.harness.state import ConversationState


def test_new_state_is_empty():
    state = ConversationState()
    assert state.messages == ()
    assert len(state) == 0


def test_add_appends_in_order():
    state = ConversationState()
    first, second = Message.user("hi"), Message.user("again")
    state.add(first)
    state.add(second)
    assert state.messages == (first, second)
    assert len(state) == 2


def test_messages_is_a_tuple():
    state = ConversationState([Message.user("hi")])
    assert isinstance(state.messages, tuple)


def test_snapshot_does_not_change_after_later_add():
    state = ConversationState()
    state.add(Message.user("one"))
    snapshot = state.messages
    state.add(Message.user("two"))
    assert len(snapshot) == 1  # the earlier snapshot is untouched
    assert len(state.messages) == 2


def test_constructor_copies_the_list():
    original = [Message.user("hi")]
    state = ConversationState(original)
    original.append(Message.user("sneaky"))  # mutate the caller's list afterwards
    assert len(state.messages) == 1


def test_add_does_not_change_the_callers_list():
    original = [Message.user("hi")]
    state = ConversationState(original)
    state.add(Message.user("more"))
    assert len(original) == 1


def test_constructor_keeps_initial_messages():
    first = Message.user("hi")
    state = ConversationState([first])
    assert state.messages == (first,)


def test_empty_list_given_is_still_a_fresh_state():
    state = ConversationState([])
    state.add(Message.user("hi"))
    assert len(state) == 1
    
def test_last_n_default_returns_last_message():
    state = ConversationState()
    state.add(Message.user("first"))
    state.add(Message.user("second"))
    result = state.last_n()
    assert len(result)  == 1 
    assert result[0].content == "second"
    
def test_last_n_with_custom_n():
    state = ConversationState()
    state.add(Message.user("a"))
    state.add(Message.user("b"))
    state.add(Message.user("c"))
    result = state.last_n(2)
    assert len(result) == 2
    assert [m.content for m in result] == ["b", "c"]
    
def test_last_n_on_empty_state_returns_none():
    assert ConversationState().last_n() is None
    
def test_last_n_larger_than_total_returns_all():
    state = ConversationState([Message.user("only")])
    result = state.last_n(5)
    assert len(result) == 1
    assert result[0].content == "only"
    
def test_last_n_return_type_is_list():
    state = ConversationState([Message.user("hi")])
    assert isinstance(state.last_n(), list)
