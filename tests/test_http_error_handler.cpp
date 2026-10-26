#include "serve/http_server.h"

#include <nlohmann/json.hpp>
#include <spdlog/logger.h>

#include <iostream>
#include <memory>
#include <string>

namespace {

using Json = nlohmann::json;
using ninfer::serve::ServeOptions;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    ServeOptions options;
    options.max_request_bytes = 1234;

    const ninfer::serve::ApiError media_budget = ninfer::serve::request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::MediaBudgetExceeded,
                             "vision tokens exceed processor budget"));
    failures += check(media_budget.status == 400 && media_budget.code == "media_budget_exceeded",
                      "media resource rejection did not map to HTTP 400");
    const ninfer::serve::ApiError invalid_media = ninfer::serve::request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::InvalidMedia, "failed to open media"));
    failures += check(invalid_media.status == 400 && invalid_media.code == "invalid_media" &&
                          invalid_media.param == "messages",
                      "invalid media did not retain its client-input classification");
    const ninfer::serve::ApiError context_limit = ninfer::serve::request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::ContextLengthExceeded,
                             "prepared prompt has 200 tokens, exceeding Engine max_context 128"));
    failures +=
        check(context_limit.status == 400 && context_limit.code == "context_length_exceeded" &&
                  context_limit.message.find("200 tokens") != std::string::npos &&
                  context_limit.message.find("128") != std::string::npos,
              "context rejection lost its HTTP classification or capacity details");
    const ninfer::serve::ApiError cancelled =
        ninfer::serve::request_error_to_api_error(ninfer::RequestError(
            ninfer::RequestErrorKind::Cancelled, "request cancelled during preparation"));
    failures += check(cancelled.status == 499 && cancelled.code == "client_disconnected" &&
                          cancelled.param.empty(),
                      "preparation cancellation did not retain its HTTP classification");

    failures +=
        check(ninfer::serve::matches_bearer_credential("Bearer secret", "secret") &&
                  ninfer::serve::matches_bearer_credential("bearer secret", "secret") &&
                  ninfer::serve::matches_bearer_credential("\tBEARER   secret\t", "secret"),
              "valid Bearer credentials were rejected because of scheme case or whitespace");
    failures += check(!ninfer::serve::matches_bearer_credential("Basic secret", "secret") &&
                          !ninfer::serve::matches_bearer_credential("Bearer wrong", "secret") &&
                          !ninfer::serve::matches_bearer_credential("Bearersecret", "secret") &&
                          !ninfer::serve::matches_bearer_credential("Bearer secret", ""),
                      "invalid Bearer credentials were accepted");
    failures += check(!ninfer::serve::matches_bearer_credential("Bearer secre", "secret") &&
                          !ninfer::serve::matches_bearer_credential("Bearer secrets", "secret"),
                      "a Bearer credential that is a prefix or extension of the key was accepted");

    // The Host and Origin checks, ahead of every route: a 403 in the endpoint's error shape.
    {
        ServeOptions local;
        const ninfer::serve::HostPolicy policy =
            ninfer::serve::make_host_policy(local.host, local.port, local.allowed_hosts, "gpubox");
        const auto refused = [&](const ServeOptions& serve, std::string path, std::string host,
                                 std::string origin, httplib::Response& response) {
            httplib::Request request;
            request.path = std::move(path);
            if (!host.empty()) { request.set_header("Host", host); }
            if (!origin.empty()) { request.set_header("Origin", origin); }
            return ninfer::serve::reject_untrusted_request(policy, serve, request, response);
        };

        httplib::Response rebound;
        failures += check(refused(local, "/v1/chat/completions", "evil.example:8080", "", rebound),
                          "a DNS-rebinding Host was not refused");
        const Json rebound_body = Json::parse(rebound.body);
        failures += check(rebound.status == 403 &&
                              rebound_body.at("error").at("code") == "host_not_allowed" &&
                              rebound_body.at("error").at("message").get<std::string>().find(
                                  "--allowed-host") != std::string::npos &&
                              rebound.get_header_value("x-request-id").starts_with("req_"),
                          "a refused Host did not get an OpenAI-shaped 403 naming --allowed-host");

        httplib::Response binary_host;
        failures += check(refused(local, "/v1/models", "\xff\xfe.example", "", binary_host) &&
                              binary_host.status == 403 &&
                              Json::parse(binary_host.body)
                                      .at("error")
                                      .at("message")
                                      .get<std::string>()
                                      .starts_with("Host '??.example'"),
                          "a Host that is not UTF-8 did not get a readable 403");

        httplib::Response rebound_messages;
        failures += check(refused(local, "/v1/messages", "evil.example", "", rebound_messages),
                          "a DNS-rebinding Host on /v1/messages was not refused");
        const Json messages_body = Json::parse(rebound_messages.body);
        failures += check(rebound_messages.status == 403 && messages_body.at("type") == "error" &&
                              messages_body.at("error").at("type") == "permission_error",
                          "a refused Host on /v1/messages did not get an Anthropic-shaped 403");

        httplib::Response health_host;
        failures += check(refused(local, "/health", "evil.example", "", health_host) &&
                              health_host.status == 403,
                          "the Host check did not apply on /health");

        // With a key the Host check is skipped: any name a client uses (a Docker, mesh VPN or
        // proxy name) reaches the server, and a rebound page still cannot send the key.
        httplib::Response keyed_host;
        ServeOptions keyed = local;
        keyed.api_key      = "secret";
        failures += check(!refused(keyed, "/health", "evil.example", "", keyed_host) &&
                              !refused(keyed, "/v1/models", "gpubox.tailnet.ts.net", "",
                                       keyed_host) &&
                              keyed_host.status == -1,
                          "the Host check applied on a server with a key");

        httplib::Response foreign;
        failures += check(refused(local, "/v1/responses", "127.0.0.1:8080",
                                  "http://evil.example", foreign),
                          "a foreign Origin was not refused without a key");
        const Json foreign_body = Json::parse(foreign.body);
        failures += check(foreign.status == 403 &&
                              foreign_body.at("error").at("code") == "origin_not_allowed",
                          "a refused Origin did not get an OpenAI-shaped 403");

        ServeOptions cors = local;
        cors.enable_cors  = true;
        httplib::Response allowed;
        failures += check(!refused(keyed, "/v1/responses", "localhost:8080", "http://evil.example",
                                   allowed) &&
                              !refused(cors, "/v1/responses", "localhost:8080",
                                       "http://evil.example", allowed) &&
                              !refused(local, "/v1/responses", "localhost:8080",
                                       "http://localhost:8080", allowed) &&
                              !refused(local, "/v1/responses", "192.168.1.5:8080", "", allowed) &&
                              !refused(local, "/v1/responses", "192.168.1.5:8080",
                                       "http://192.168.1.5:8080", allowed) &&
                              !refused(local, "/v1/responses", "localhost:8080",
                                       "chrome-extension://abcdef", allowed) &&
                              !refused(local, "/v1/responses", "", "", allowed) &&
                              allowed.body.empty() && allowed.status == -1,
                          "an allowed request was refused or given a response");
    }

    httplib::Request messages_request;
    messages_request.path = "/v1/messages";
    httplib::Response messages_response;
    messages_response.status = 400;
    const auto initial_messages_result =
        ninfer::serve::handle_unrendered_http_error(options, messages_request, messages_response);
    const std::string messages_request_id = messages_response.get_header_value("request-id");
    failures += check(initial_messages_result == httplib::Server::HandlerResponse::Unhandled &&
                          messages_response.body.empty() && messages_request_id.starts_with("req_"),
                      "Anthropic HTTP request ID was not established before error rendering");
    messages_response.status = 413;
    const auto messages_result =
        ninfer::serve::handle_unrendered_http_error(options, messages_request, messages_response);
    const Json messages_body = Json::parse(messages_response.body);
    failures += check(messages_result == httplib::Server::HandlerResponse::Handled &&
                          messages_body.at("type") == "error" &&
                          messages_body.at("error").at("type") == "request_too_large" &&
                          messages_body.at("request_id").get<std::string>().starts_with("req_") &&
                          messages_body.at("request_id") == messages_request_id &&
                          messages_response.get_header_value_count("request-id") == 1 &&
                          messages_response.get_header_value("request-id") ==
                              messages_body.at("request_id").get<std::string>() &&
                          !messages_response.has_header("x-request-id") &&
                          messages_body.at("error").at("message").get<std::string>().find(
                              "1234 bytes") != std::string::npos,
                      "empty Anthropic 413 did not become a payload-limit error");

    httplib::Request openai_request;
    openai_request.path = "/v1/responses";
    httplib::Response openai_response;
    openai_response.status = 413;
    const auto openai_result =
        ninfer::serve::handle_unrendered_http_error(options, openai_request, openai_response);
    const Json openai_body = Json::parse(openai_response.body);
    failures += check(openai_result == httplib::Server::HandlerResponse::Handled &&
                          openai_body.at("error").at("code") == "request_too_large" &&
                          openai_response.get_header_value_count("x-request-id") == 1 &&
                          openai_response.get_header_value("x-request-id").starts_with("req_") &&
                          openai_body.at("error").at("message").get<std::string>().find(
                              "1234 bytes") != std::string::npos,
                      "empty OpenAI 413 did not become a payload-limit error");

    httplib::Request missing_messages_request;
    missing_messages_request.path = "/v1/messages/missing";
    httplib::Response missing_messages_response;
    missing_messages_response.status = 404;
    missing_messages_response.set_header("request-id", "req_existing");
    const auto missing_messages_result = ninfer::serve::handle_unrendered_http_error(
        options, missing_messages_request, missing_messages_response);
    const Json missing_messages_body = Json::parse(missing_messages_response.body);
    failures +=
        check(missing_messages_result == httplib::Server::HandlerResponse::Handled &&
                  missing_messages_response.status == 404 &&
                  missing_messages_body.at("error").at("type") == "not_found_error" &&
                  missing_messages_body.at("request_id").get<std::string>().starts_with("req_") &&
                  missing_messages_body.at("request_id") == "req_existing" &&
                  missing_messages_response.get_header_value_count("request-id") == 1 &&
                  missing_messages_response.get_header_value("request-id") ==
                      missing_messages_body.at("request_id").get<std::string>(),
              "missing Anthropic resource did not use the protocol error envelope");

    httplib::Response authored_messages_response;
    authored_messages_response.status = 413;
    authored_messages_response.set_header("request-id", "req_authored");
    authored_messages_response.set_content(
        R"({"type":"error","error":{"type":"request_too_large"},"request_id":"req_authored"})",
        "application/json");
    const std::string authored_messages_body = authored_messages_response.body;
    const auto authored_messages_result      = ninfer::serve::handle_unrendered_http_error(
        options, messages_request, authored_messages_response);
    failures +=
        check(authored_messages_result == httplib::Server::HandlerResponse::Unhandled &&
                  authored_messages_response.body == authored_messages_body &&
                  authored_messages_response.get_header_value_count("request-id") == 1 &&
                  authored_messages_response.get_header_value("request-id") == "req_authored",
              "application-authored Anthropic error or its request ID was overwritten");

    httplib::Response authored_response;
    authored_response.status = 413;
    authored_response.set_header("x-request-id", "req_existing");
    authored_response.set_content(R"({"error":{"code":"application_error"}})", "application/json");
    const std::string authored_body = authored_response.body;
    const auto authored_result =
        ninfer::serve::handle_unrendered_http_error(options, openai_request, authored_response);
    failures += check(authored_result == httplib::Server::HandlerResponse::Unhandled &&
                          authored_response.body == authored_body &&
                          authored_response.get_header_value_count("x-request-id") == 1 &&
                          authored_response.get_header_value("x-request-id") == "req_existing",
                      "application-authored 413 or its request ID was overwritten");

    httplib::Response other_response;
    other_response.status = 400;
    const auto other_result =
        ninfer::serve::handle_unrendered_http_error(options, openai_request, other_response);
    failures += check(other_result == httplib::Server::HandlerResponse::Unhandled &&
                          other_response.body.empty(),
                      "non-413 response was changed by the payload-limit handler");

    // dashboard_while_loading: a keyed server still loading its model serves GET /dashboard
    // without the key (the page holds no data) with its CSP and nosniff headers, under any Host
    // (a keyed server skips the Host check), and still answers every other route, POST /dashboard
    // included, with the loading 503 -- and, once that is past, the key check.
    {
        ServeOptions keyed;
        keyed.host    = "127.0.0.1";
        keyed.api_key = "secret";
        ninfer::serve::HttpServer server(keyed, std::make_shared<spdlog::logger>("dashboard-test"));
        const int port = server.bind_any_port();
        failures += check(port > 0, "dashboard_while_loading: could not bind a loopback port");
        if (port > 0) {
            server.start_serving_during_startup();
            httplib::Client client("127.0.0.1", port);
            client.set_connection_timeout(5, 0);
            client.set_read_timeout(5, 0);
            const auto page = client.Get("/dashboard");
            failures += check(
                page && page->status == 200 &&
                    page->get_header_value("Content-Type") == "text/html; charset=utf-8" &&
                    page->get_header_value("X-Content-Type-Options") == "nosniff" &&
                    page->get_header_value("Content-Security-Policy").find("default-src 'none'") !=
                        std::string::npos &&
                    page->body.rfind("<!doctype html>", 0) == 0,
                "dashboard_while_loading: GET /dashboard was not served without the key");
            const auto rebound = client.Get("/dashboard", httplib::Headers{{"Host", "evil.example:8080"}});
            failures += check(rebound && rebound->status == 200,
                              "dashboard_while_loading: a keyed server ran the Host check");
            const auto metrics = client.Get("/metrics");
            const auto posted  = client.Post("/dashboard", "", "text/plain");
            failures += check(metrics && metrics->status == 503 && posted && posted->status == 503,
                              "dashboard_while_loading: another route skipped the loading 503");
            server.stop();
            (void)server.await_startup_listener();
        }
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
