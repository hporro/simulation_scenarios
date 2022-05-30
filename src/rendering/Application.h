#pragma once

#include <GL/glew.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include <vector>

#include "../logging/Logging.h"
#include "../timing/timing_helpers.h"

enum Event_IDs {
	Null = 0,
	WindowResize = 1,
	Button = 2,
};

struct Event { Event_IDs m_id; bool handled = false; Event(Event_IDs evid); };

struct WindowResizeEvent : public Event {
	WindowResizeEvent(int width, int height);
	int m_width, m_height;
};

struct ButtonEvent : public Event {
	ButtonEvent(int key, int scancode, int action, int mods);
	int m_key, m_scancode, m_action, m_mods;
};

struct Application {
	Application(int width, int height, const std::string title);
	void run();
	void setWidthHeight(int width, int height);
	void setup_callbacks();
	void dispatch_events();

	std::vector<Event*> eventQueue;
	GLFWwindow* window = nullptr;
	bool opengl_context_initialized = false;
	bool logging_initialized = false;
	bool imgui_context_initialized = false;
	int m_width = 0, m_height = 0;
};