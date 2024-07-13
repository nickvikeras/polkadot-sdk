// This file is part of Substrate.

// Copyright (C) Parity Technologies (UK) Ltd.
// SPDX-License-Identifier: Apache-2.0

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;
use frame_support::traits::{OnRuntimeUpgrade, UncheckedOnRuntimeUpgrade};
use sp_std::vec::Vec;

#[cfg(feature = "try-runtime")]
use sp_runtime::TryRuntimeError;

pub mod unversioned {
	use super::*;

	/// Migrates delegation from older derivation of [`AccountType::ProxyDelegator`] accounts
	/// to the new one for all agents.
	pub struct ProxyDelegatorMigration<T, MaxAgents>(sp_std::marker::PhantomData<(T, MaxAgents)>);

	impl<T: Config, MaxAgents: Get<u32>> OnRuntimeUpgrade for ProxyDelegatorMigration<T, MaxAgents> {
		fn on_runtime_upgrade() -> Weight {
			use sp_runtime::traits::AccountIdConversion;
			let mut success: u32 = 0;
			let mut fails: u32 = 0;
			let old_proxy_delegator = |agent: T::AccountId| {
				T::PalletId::get()
					.into_sub_account_truncating((AccountType::ProxyDelegator, agent.clone()))
			};
			Agents::<T>::iter_keys().take(MaxAgents::get() as usize).for_each(|agent| {
				let old_proxy = old_proxy_delegator(agent.clone());

				Delegation::<T>::get(&old_proxy).map(|delegation| {
					let new_proxy =
						Pallet::<T>::generate_proxy_delegator(Agent::from(agent.clone()));

					let _ = Pallet::<T>::do_migrate_delegation(
						Delegator::from(old_proxy),
						Delegator::from(new_proxy),
						delegation.amount,
					)
					.map(success += 1)
					.map_err(|e| {
						log!(
							info,
							"Failed to migrate proxy delegator for agent {:?}, old proxy delegator: {:?}",
							agent,
							old_proxy
						);

						fails += 1;
					});
				});
			});

			log!(
				info,
				"Migration to new proxy delegator account success for {:?} agents, failed for {:?} agents",
				success,
				fails
			);

			Weight::default()
		}

		#[cfg(feature = "try-runtime")]
		fn post_upgrade(data: Vec<u8>) -> Result<(), TryRuntimeError> {
			Ok(())
		}
	}
}
