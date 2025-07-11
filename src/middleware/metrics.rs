use crate::state::AppState;
use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    web, Error,
};
use futures_util::future::LocalBoxFuture;
use std::{
    future::{ready, Ready},
    time::Instant,
};

pub struct MetricsMiddleware;

impl<S, B> Transform<S, ServiceRequest> for MetricsMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = MetricsMiddlewareService<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(MetricsMiddlewareService { service }))
    }
}

pub struct MetricsMiddlewareService<S> {
    service: S,
}

impl<S, B> Service<ServiceRequest> for MetricsMiddlewareService<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let start_time = Instant::now();
        let method = req.method().to_string();
        let path = req.uri().path().to_string();
        let endpoint = format!("{} {}", method, path);

        if let Some(app_state) = req.app_data::<web::Data<AppState>>() {
            app_state.increment_request_count();
        }

        let fut = self.service.call(req);

        Box::pin(async move {
            let result = fut.await;
            let duration = start_time.elapsed();
            let duration_ms = duration.as_millis() as u64;

            let is_error = match &result {
                Ok(response) => response.status().is_client_error() || response.status().is_server_error(),
                Err(_) => true,
            };

            if let Ok(response) = &result {
                if let Some(app_state) = response.request().app_data::<web::Data<AppState>>() {
                    app_state.record_endpoint_request(&endpoint, duration_ms, is_error);
                    
                    if is_error {
                        app_state.increment_error_count();
                    }
                }
            }

            result
        })
    }
}