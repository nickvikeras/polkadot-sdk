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

//! Environment definition of the wasm smart-contract runtime.

use crate::{
	exec::{ExecError, ExecResult, Ext, Key, TopicOf},
	gas::{ChargedAmount, Token},
	primitives::ExecReturnValue,
	weights::WeightInfo,
	BalanceOf, CodeHash, Config, DebugBufferVec, Error, LOG_TARGET, SENTINEL,
};
use alloc::{boxed::Box, vec, vec::Vec};
use codec::{Decode, DecodeLimit, Encode, MaxEncodedLen};
use core::{fmt, marker::PhantomData};
use frame_support::{
	dispatch::DispatchInfo, ensure, pallet_prelude::DispatchResultWithPostInfo, parameter_types,
	traits::Get, weights::Weight,
};
use pallet_contracts_proc_macro::define_env;
use pallet_contracts_uapi::{CallFlags, ReturnFlags};
use sp_io::hashing::{blake2_128, blake2_256, keccak_256, sha2_256};
use sp_runtime::{traits::Zero, DispatchError, RuntimeDebug};

type CallOf<T> = <T as frame_system::Config>::RuntimeCall;

/// The maximum nesting depth a contract can use when encoding types.
const MAX_DECODE_NESTING: u32 = 256;

#[derive(Clone, Copy)]
pub enum ApiVersion {
	/// Expose all APIs even unversioned ones. Only used for testing and benchmarking.
	UnsafeNewest,
	/// Only expose API's up to and including the specified version.
	Versioned(u16),
}

/// Abstraction over the memory access within syscalls.
///
/// The reason for this abstraction is that we run syscalls on the host machine when
/// benchmarking them. In that case we have direct access to the contracts memory. However, when
/// running within PolkaVM we need to resort to copying as we can't map the contracts memory into
/// the host (as of now).
pub trait Memory<T: Config> {
	/// Read designated chunk from the sandbox memory into the supplied buffer.
	///
	/// Returns `Err` if one of the following conditions occurs:
	///
	/// - requested buffer is not within the bounds of the sandbox memory.
	fn read_into_buf(&self, ptr: u32, buf: &mut [u8]) -> Result<(), DispatchError>;

	/// Write the given buffer to the designated location in the sandbox memory.
	///
	/// Returns `Err` if one of the following conditions occurs:
	///
	/// - designated area is not within the bounds of the sandbox memory.
	fn write(&mut self, ptr: u32, buf: &[u8]) -> Result<(), DispatchError>;

	/// Read designated chunk from the sandbox memory.
	///
	/// Returns `Err` if one of the following conditions occurs:
	///
	/// - requested buffer is not within the bounds of the sandbox memory.
	fn read(&self, ptr: u32, len: u32) -> Result<Vec<u8>, DispatchError> {
		let mut buf = vec![0u8; len as usize];
		self.read_into_buf(ptr, buf.as_mut_slice())?;
		Ok(buf)
	}

	/// Read designated chunk from the sandbox memory and attempt to decode into the specified type.
	///
	/// Returns `Err` if one of the following conditions occurs:
	///
	/// - requested buffer is not within the bounds of the sandbox memory.
	/// - the buffer contents cannot be decoded as the required type.
	///
	/// # Note
	///
	/// There must be an extra benchmark for determining the influence of `len` with
	/// regard to the overall weight.
	fn read_as_unbounded<D: Decode>(&self, ptr: u32, len: u32) -> Result<D, DispatchError> {
		let buf = self.read(ptr, len)?;
		let decoded = D::decode_all_with_depth_limit(MAX_DECODE_NESTING, &mut buf.as_ref())
			.map_err(|_| DispatchError::from(Error::<T>::DecodingFailed))?;
		Ok(decoded)
	}

	/// Reads and decodes a type with a size fixed at compile time from contract memory.
	///
	/// # Only use on fixed size types
	///
	/// Don't use this for types where the encoded size is not fixed but merely bounded. Otherwise
	/// this implementation will out of bound access the buffer declared by the guest. Some examples
	/// of those bounded but not fixed types: Enums with data, `BoundedVec` or any compact encoded
	/// integer.
	///
	/// # Note
	///
	/// The weight of reading a fixed value is included in the overall weight of any
	/// contract callable function.
	fn read_as<D: Decode + MaxEncodedLen>(&self, ptr: u32) -> Result<D, DispatchError> {
		debug_assert_eq!(
			D::encoded_fixed_size(),
			Some(D::max_encoded_len()),
			"Memory::read_as() can only be called on fixed size types. Use read_as_unbounded instead."
		);
		let buf = self.read(ptr, D::max_encoded_len() as u32)?;
		let decoded = D::decode_with_depth_limit(MAX_DECODE_NESTING, &mut buf.as_ref())
			.map_err(|_| DispatchError::from(Error::<T>::DecodingFailed))?;
		Ok(decoded)
	}
}

/// Allows syscalls access to the PolkaVM instance they are executing in.
///
/// In case a contract is executing within PolkaVM its `memory` argument will also implement
/// this trait. The benchmarking implementation of syscalls will only require `Memory`
/// to be implemented.
trait PolkavmInstance<T: Config>: Memory<T> {
	fn gas(&self) -> polkavm::Gas;
	fn set_gas(&mut self, gas: polkavm::Gas);
	fn read_input_regs(&self) -> (u32, u32, u32, u32, u32, u32);
}

// Memory implementation used in benchmarking where guest memory is mapped into the host.
//
// Please note that we could optimize the `read_as_*` functions by decoding directly from
// memory without a copy. However, we don't do that because as it would change the behaviour
// of those functions: A `read_as` with a `len` larger than the actual type can succeed
// in the streaming implementation while it could fail with a segfault in the copy implementation.
#[cfg(feature = "runtime-benchmarks")]
pub struct MappedMemory<'a, T> {
	memory: &'a mut [u8],
	phantom: PhantomData<T>,
}

#[cfg(feature = "runtime-benchmarks")]
impl<'a, T> MappedMemory<'a, T> {
	pub fn new(memory: &'a mut [u8]) -> Self {
		Self { memory, phantom: Default::default() }
	}
}

#[cfg(feature = "runtime-benchmarks")]
impl<'a, T: Config> Memory<T> for MappedMemory<'a, T> {
	fn read_into_buf(&self, ptr: u32, buf: &mut [u8]) -> Result<(), DispatchError> {
		let ptr = ptr as usize;
		let bound_checked =
			self.memory.get(ptr..ptr + buf.len()).ok_or_else(|| Error::<T>::OutOfBounds)?;
		buf.copy_from_slice(bound_checked);
		Ok(())
	}

	fn write(&mut self, ptr: u32, buf: &[u8]) -> Result<(), DispatchError> {
		let ptr = ptr as usize;
		let bound_checked = self
			.memory
			.get_mut(ptr..ptr + buf.len())
			.ok_or_else(|| Error::<T>::OutOfBounds)?;
		bound_checked.copy_from_slice(buf);
		Ok(())
	}
}

/// Memory implementation used for PolkaVM where guest memory has to be copied for access.
pub struct PolkavmMemory<'a, T: Config> {
	instance: &'a mut polkavm::RawInstance,
	_phantom: PhantomData<T>,
}

impl<'a, T: Config> PolkavmMemory<'a, T> {
	pub fn new(instance: &'a mut polkavm::RawInstance) -> Self {
		Self { instance, _phantom: Default::default() }
	}
}

impl<'a, T: Config> Memory<T> for PolkavmMemory<'a, T> {
	fn read_into_buf(&self, ptr: u32, buf: &mut [u8]) -> Result<(), DispatchError> {
		self.instance
			.read_memory_into(ptr, buf)
			.map(|_| ())
			.map_err(|_| Error::<T>::OutOfBounds.into())
	}

	fn write(&mut self, ptr: u32, buf: &[u8]) -> Result<(), DispatchError> {
		self.instance.write_memory(ptr, buf).map_err(|_| Error::<T>::OutOfBounds.into())
	}
}

impl<'a, T: Config> PolkavmInstance<T> for PolkavmMemory<'a, T> {
	fn gas(&self) -> polkavm::Gas {
		self.instance.gas()
	}

	fn set_gas(&mut self, gas: polkavm::Gas) {
		self.instance.set_gas(gas)
	}

	fn read_input_regs(&self) -> (u32, u32, u32, u32, u32, u32) {
		(
			self.instance.reg(polkavm::Reg::A0),
			self.instance.reg(polkavm::Reg::A1),
			self.instance.reg(polkavm::Reg::A2),
			self.instance.reg(polkavm::Reg::A3),
			self.instance.reg(polkavm::Reg::A4),
			self.instance.reg(polkavm::Reg::A5),
		)
	}
}

/// Type of a storage key.
enum KeyType {
	/// Legacy fix sized key `[u8;32]`.
	Fix,
	/// Variable sized key used in transparent hashing,
	/// cannot be larger than MaxStorageKeyLen.
	Var(u32),
}

pub use pallet_contracts_uapi::ReturnErrorCode;

parameter_types! {
	/// Getter types used by [`crate::api_doc::Current::call_runtime`]
	const CallRuntimeFailed: ReturnErrorCode = ReturnErrorCode::CallRuntimeFailed;
	/// Getter types used by [`crate::api_doc::Current::xcm_execute`]
	const XcmExecutionFailed: ReturnErrorCode = ReturnErrorCode::XcmExecutionFailed;
}

impl From<ExecReturnValue> for ReturnErrorCode {
	fn from(from: ExecReturnValue) -> Self {
		if from.flags.contains(ReturnFlags::REVERT) {
			Self::CalleeReverted
		} else {
			Self::Success
		}
	}
}

/// The data passed through when a contract uses `seal_return`.
#[derive(RuntimeDebug)]
pub struct ReturnData {
	/// The flags as passed through by the contract. They are still unchecked and
	/// will later be parsed into a `ReturnFlags` bitflags struct.
	flags: u32,
	/// The output buffer passed by the contract as return data.
	data: Vec<u8>,
}

/// Enumerates all possible reasons why a trap was generated.
///
/// This is either used to supply the caller with more information about why an error
/// occurred (the SupervisorError variant).
/// The other case is where the trap does not constitute an error but rather was invoked
/// as a quick way to terminate the application (all other variants).
#[derive(RuntimeDebug)]
pub enum TrapReason {
	/// The supervisor trapped the contract because of an error condition occurred during
	/// execution in privileged code.
	SupervisorError(DispatchError),
	/// Signals that trap was generated in response to call `seal_return` host function.
	Return(ReturnData),
	/// Signals that a trap was generated in response to a successful call to the
	/// `seal_terminate` host function.
	Termination,
}

impl<T: Into<DispatchError>> From<T> for TrapReason {
	fn from(from: T) -> Self {
		Self::SupervisorError(from.into())
	}
}

impl fmt::Display for TrapReason {
	fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
		Ok(())
	}
}

#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
#[derive(Copy, Clone)]
pub enum RuntimeCosts {
	/// Base Weight of calling a host function.
	HostFn,
	/// Weight charged for copying data from the sandbox.
	CopyFromContract(u32),
	/// Weight charged for copying data to the sandbox.
	CopyToContract(u32),
	/// Weight of calling `seal_caller`.
	Caller,
	/// Weight of calling `seal_is_contract`.
	IsContract,
	/// Weight of calling `seal_code_hash`.
	CodeHash,
	/// Weight of calling `seal_own_code_hash`.
	OwnCodeHash,
	/// Weight of calling `seal_caller_is_origin`.
	CallerIsOrigin,
	/// Weight of calling `caller_is_root`.
	CallerIsRoot,
	/// Weight of calling `seal_address`.
	Address,
	/// Weight of calling `seal_gas_left`.
	GasLeft,
	/// Weight of calling `seal_balance`.
	Balance,
	/// Weight of calling `seal_value_transferred`.
	ValueTransferred,
	/// Weight of calling `seal_minimum_balance`.
	MinimumBalance,
	/// Weight of calling `seal_block_number`.
	BlockNumber,
	/// Weight of calling `seal_now`.
	Now,
	/// Weight of calling `seal_weight_to_fee`.
	WeightToFee,
	/// Weight of calling `seal_terminate`, passing the number of locked dependencies.
	Terminate(u32),
	/// Weight of calling `seal_deposit_event` with the given number of topics and event size.
	DepositEvent { num_topic: u32, len: u32 },
	/// Weight of calling `seal_debug_message` per byte of passed message.
	DebugMessage(u32),
	/// Weight of calling `seal_set_storage` for the given storage item sizes.
	SetStorage { old_bytes: u32, new_bytes: u32 },
	/// Weight of calling `seal_clear_storage` per cleared byte.
	ClearStorage(u32),
	/// Weight of calling `seal_contains_storage` per byte of the checked item.
	ContainsStorage(u32),
	/// Weight of calling `seal_get_storage` with the specified size in storage.
	GetStorage(u32),
	/// Weight of calling `seal_take_storage` for the given size.
	TakeStorage(u32),
	/// Weight of calling `seal_set_transient_storage` for the given storage item sizes.
	SetTransientStorage { old_bytes: u32, new_bytes: u32 },
	/// Weight of calling `seal_clear_transient_storage` per cleared byte.
	ClearTransientStorage(u32),
	/// Weight of calling `seal_contains_transient_storage` per byte of the checked item.
	ContainsTransientStorage(u32),
	/// Weight of calling `seal_get_transient_storage` with the specified size in storage.
	GetTransientStorage(u32),
	/// Weight of calling `seal_take_transient_storage` for the given size.
	TakeTransientStorage(u32),
	/// Weight of calling `seal_transfer`.
	Transfer,
	/// Base weight of calling `seal_call`.
	CallBase,
	/// Weight of calling `seal_delegate_call` for the given input size.
	DelegateCallBase,
	/// Weight of the transfer performed during a call.
	CallTransferSurcharge,
	/// Weight per byte that is cloned by supplying the `CLONE_INPUT` flag.
	CallInputCloned(u32),
	/// Weight of calling `seal_instantiate` for the given input length and salt.
	Instantiate { input_data_len: u32, salt_len: u32 },
	/// Weight of calling `seal_hash_sha_256` for the given input size.
	HashSha256(u32),
	/// Weight of calling `seal_hash_keccak_256` for the given input size.
	HashKeccak256(u32),
	/// Weight of calling `seal_hash_blake2_256` for the given input size.
	HashBlake256(u32),
	/// Weight of calling `seal_hash_blake2_128` for the given input size.
	HashBlake128(u32),
	/// Weight of calling `seal_ecdsa_recover`.
	EcdsaRecovery,
	/// Weight of calling `seal_sr25519_verify` for the given input size.
	Sr25519Verify(u32),
	/// Weight charged by a chain extension through `seal_call_chain_extension`.
	ChainExtension(Weight),
	/// Weight charged for calling into the runtime.
	CallRuntime(Weight),
	/// Weight charged for calling xcm_execute.
	CallXcmExecute(Weight),
	/// Weight of calling `seal_set_code_hash`
	SetCodeHash,
	/// Weight of calling `ecdsa_to_eth_address`
	EcdsaToEthAddress,
	/// Weight of calling `lock_delegate_dependency`
	LockDelegateDependency,
	/// Weight of calling `unlock_delegate_dependency`
	UnlockDelegateDependency,
}

// For the function that modifies the storage, the benchmarks are done with one item in the
// transient_storage (BTreeMap). To consider the worst-case scenario, the weight of the overhead of
// writing to a full BTreeMap should be included. On top of that, the rollback weight is added,
// which is the worst scenario.
macro_rules! cost_write {
	// cost_write!(name, a, b, c) -> T::WeightInfo::name(a, b, c).saturating_add(T::WeightInfo::rollback_transient_storage())
	// .saturating_add(T::WeightInfo::set_transient_storage_full().saturating_sub(T::WeightInfo::set_transient_storage_empty())
	($name:ident $(, $arg:expr )*) => {
		(T::WeightInfo::$name($( $arg ),*).saturating_add(T::WeightInfo::rollback_transient_storage()).saturating_add(cost_write!(@cost_storage)))
	};

	(@cost_storage) => {
        T::WeightInfo::set_transient_storage_full().saturating_sub(T::WeightInfo::set_transient_storage_empty())
    };
}

macro_rules! cost_read {
	// cost_read!(name, a, b, c) -> T::WeightInfo::name(a, b, c).saturating_add(T::WeightInfo::get_transient_storage_full()
	// .saturating_sub(T::WeightInfo::get_transient_storage_empty())
	($name:ident $(, $arg:expr )*) => {
		(T::WeightInfo::$name($( $arg ),*).saturating_add(cost_read!(@cost_storage)))
	};

	(@cost_storage) => {
        T::WeightInfo::get_transient_storage_full().saturating_sub(T::WeightInfo::get_transient_storage_empty())
    };
}

macro_rules! cost_args {
	// cost_args!(name, a, b, c) -> T::WeightInfo::name(a, b, c).saturating_sub(T::WeightInfo::name(0, 0, 0))
	($name:ident, $( $arg: expr ),+) => {
		(T::WeightInfo::$name($( $arg ),+).saturating_sub(cost_args!(@call_zero $name, $( $arg ),+)))
	};
	// Transform T::WeightInfo::name(a, b, c) into T::WeightInfo::name(0, 0, 0)
	(@call_zero $name:ident, $( $arg:expr ),*) => {
		T::WeightInfo::$name($( cost_args!(@replace_token $arg) ),*)
	};
	// Replace the token with 0.
	(@replace_token $_in:tt) => { 0 };
}

impl<T: Config> Token<T> for RuntimeCosts {
	fn influence_lowest_gas_limit(&self) -> bool {
		match self {
			&Self::CallXcmExecute(_) => false,
			_ => true,
		}
	}

	fn weight(&self) -> Weight {
		use self::RuntimeCosts::*;
		match *self {
			HostFn => cost_args!(noop_host_fn, 1),
			CopyToContract(len) => T::WeightInfo::seal_input(len),
			CopyFromContract(len) => T::WeightInfo::seal_return(len),
			Caller => T::WeightInfo::seal_caller(),
			IsContract => T::WeightInfo::seal_is_contract(),
			CodeHash => T::WeightInfo::seal_code_hash(),
			OwnCodeHash => T::WeightInfo::seal_own_code_hash(),
			CallerIsOrigin => T::WeightInfo::seal_caller_is_origin(),
			CallerIsRoot => T::WeightInfo::seal_caller_is_root(),
			Address => T::WeightInfo::seal_address(),
			GasLeft => T::WeightInfo::seal_gas_left(),
			Balance => T::WeightInfo::seal_balance(),
			ValueTransferred => T::WeightInfo::seal_value_transferred(),
			MinimumBalance => T::WeightInfo::seal_minimum_balance(),
			BlockNumber => T::WeightInfo::seal_block_number(),
			Now => T::WeightInfo::seal_now(),
			WeightToFee => T::WeightInfo::seal_weight_to_fee(),
			Terminate(locked_dependencies) => T::WeightInfo::seal_terminate(locked_dependencies),
			DepositEvent { num_topic, len } => T::WeightInfo::seal_deposit_event(num_topic, len),
			DebugMessage(len) => T::WeightInfo::seal_debug_message(len),
			SetStorage { new_bytes, old_bytes } =>
				T::WeightInfo::seal_set_storage(new_bytes, old_bytes),
			ClearStorage(len) => T::WeightInfo::seal_clear_storage(len),
			ContainsStorage(len) => T::WeightInfo::seal_contains_storage(len),
			GetStorage(len) => T::WeightInfo::seal_get_storage(len),
			TakeStorage(len) => T::WeightInfo::seal_take_storage(len),
			SetTransientStorage { new_bytes, old_bytes } =>
				cost_write!(seal_set_transient_storage, new_bytes, old_bytes),
			ClearTransientStorage(len) => cost_write!(seal_clear_transient_storage, len),
			ContainsTransientStorage(len) => cost_read!(seal_contains_transient_storage, len),
			GetTransientStorage(len) => cost_read!(seal_get_transient_storage, len),
			TakeTransientStorage(len) => cost_write!(seal_take_transient_storage, len),
			Transfer => T::WeightInfo::seal_transfer(),
			CallBase => T::WeightInfo::seal_call(0, 0),
			DelegateCallBase => T::WeightInfo::seal_delegate_call(),
			CallTransferSurcharge => cost_args!(seal_call, 1, 0),
			CallInputCloned(len) => cost_args!(seal_call, 0, len),
			Instantiate { input_data_len, salt_len } =>
				T::WeightInfo::seal_instantiate(input_data_len, salt_len),
			HashSha256(len) => T::WeightInfo::seal_hash_sha2_256(len),
			HashKeccak256(len) => T::WeightInfo::seal_hash_keccak_256(len),
			HashBlake256(len) => T::WeightInfo::seal_hash_blake2_256(len),
			HashBlake128(len) => T::WeightInfo::seal_hash_blake2_128(len),
			EcdsaRecovery => T::WeightInfo::seal_ecdsa_recover(),
			Sr25519Verify(len) => T::WeightInfo::seal_sr25519_verify(len),
			ChainExtension(weight) | CallRuntime(weight) | CallXcmExecute(weight) => weight,
			SetCodeHash => T::WeightInfo::seal_set_code_hash(),
			EcdsaToEthAddress => T::WeightInfo::seal_ecdsa_to_eth_address(),
			LockDelegateDependency => T::WeightInfo::lock_delegate_dependency(),
			UnlockDelegateDependency => T::WeightInfo::unlock_delegate_dependency(),
		}
	}
}

/// Same as [`Runtime::charge_gas`].
///
/// We need this access as a macro because sometimes hiding the lifetimes behind
/// a function won't work out.
macro_rules! charge_gas {
	($runtime:expr, $costs:expr) => {{
		$runtime.ext.gas_meter_mut().charge($costs)
	}};
}

/// The kind of call that should be performed.
enum CallType {
	/// Execute another instantiated contract
	Call { callee_ptr: u32, value_ptr: u32, deposit_ptr: u32, weight: Weight },
	/// Execute deployed code in the context (storage, account ID, value) of the caller contract
	DelegateCall { code_hash_ptr: u32 },
}

impl CallType {
	fn cost(&self) -> RuntimeCosts {
		match self {
			CallType::Call { .. } => RuntimeCosts::CallBase,
			CallType::DelegateCall { .. } => RuntimeCosts::DelegateCallBase,
		}
	}
}

/// This is only appropriate when writing out data of constant size that does not depend on user
/// input. In this case the costs for this copy was already charged as part of the token at
/// the beginning of the API entry point.
fn already_charged(_: u32) -> Option<RuntimeCosts> {
	None
}

/// Can only be used for one call.
pub struct Runtime<'a, E: Ext + 'a> {
	ext: &'a mut E,
	input_data: Option<Vec<u8>>,
	chain_extension: Option<Box<<E::T as Config>::ChainExtension>>,
}

impl<'a, E: Ext + 'a> Runtime<'a, E> {
	pub fn new(ext: &'a mut E, input_data: Vec<u8>) -> Self {
		Runtime {
			ext,
			input_data: Some(input_data),
			chain_extension: Some(Box::new(Default::default())),
		}
	}

	pub fn handle_interrupt(
		&mut self,
		interrupt: Result<polkavm::InterruptKind, polkavm::Error>,
		instance: &mut polkavm::RawInstance,
		api_version: ApiVersion,
	) -> Option<ExecResult> {
		use polkavm::InterruptKind::*;

		match interrupt {
			Err(error) => {
				// in contrast to the other returns this "should" not happen: log level error
				log::error!(target: LOG_TARGET, "polkavm execution error: {error}");
				Some(Err(Error::<E::T>::ContractTrapped.into()))
			},
			Ok(Finished) =>
				Some(Ok(ExecReturnValue { flags: ReturnFlags::empty(), data: Vec::new() })),
			Ok(Trap(trap)) => {
				// TODO: custom error
				log::debug!(target: LOG_TARGET, "polkavm trap: {trap:?}");
				Some(Err(Error::<E::T>::ContractTrapped.into()))
			},
			Ok(Segfault(segfault)) => {
				// TODO: custom error
				log::debug!(target: LOG_TARGET, "polkavm segfault: {segfault:?}");
				Some(Err(Error::<E::T>::ContractTrapped.into()))
			},
			Ok(NotEnoughGas) => Some(Err(Error::<E::T>::OutOfGas.into())),
			Ok(Step) => None,
			Ok(Ecalli(syscall_number)) => {
				let mut memory = PolkavmMemory::new(instance);
				match self.handle_ecall(&mut memory, syscall_number, api_version) {
					Ok(None) => None,
					Ok(Some(return_value)) => {
						instance.set_reg(polkavm::Reg::A0, return_value);
						None
					},
					Err(TrapReason::Return(ReturnData { flags, data })) =>
						match ReturnFlags::from_bits(flags) {
							None => Some(Err(Error::<E::T>::InvalidCallFlags.into())),
							Some(flags) => Some(Ok(ExecReturnValue { flags, data })),
						},
					Err(TrapReason::Termination) =>
						Some(Ok(ExecReturnValue { flags: ReturnFlags::empty(), data: Vec::new() })),
					Err(TrapReason::SupervisorError(error)) => Some(Err(error.into())),
				}
			},
		}
	}

	/// Get a mutable reference to the inner `Ext`.
	///
	/// This is mainly for the chain extension to have access to the environment the
	/// contract is executing in.
	pub fn ext(&mut self) -> &mut E {
		self.ext
	}

	/// Charge the gas meter with the specified token.
	///
	/// Returns `Err(HostError)` if there is not enough gas.
	pub fn charge_gas(&mut self, costs: RuntimeCosts) -> Result<ChargedAmount, DispatchError> {
		charge_gas!(self, costs)
	}

	/// Adjust a previously charged amount down to its actual amount.
	///
	/// This is when a maximum a priori amount was charged and then should be partially
	/// refunded to match the actual amount.
	pub fn adjust_gas(&mut self, charged: ChargedAmount, actual_costs: RuntimeCosts) {
		self.ext.gas_meter_mut().adjust_gas(charged, actual_costs);
	}

	/// Charge, Run and adjust gas, for executing the given dispatchable.
	fn call_dispatchable<ErrorReturnCode: Get<ReturnErrorCode>>(
		&mut self,
		dispatch_info: DispatchInfo,
		runtime_cost: impl Fn(Weight) -> RuntimeCosts,
		run: impl FnOnce(&mut Self) -> DispatchResultWithPostInfo,
	) -> Result<ReturnErrorCode, TrapReason> {
		use frame_support::dispatch::extract_actual_weight;
		let charged = self.charge_gas(runtime_cost(dispatch_info.weight))?;
		let result = run(self);
		let actual_weight = extract_actual_weight(&result, &dispatch_info);
		self.adjust_gas(charged, runtime_cost(actual_weight));
		match result {
			Ok(_) => Ok(ReturnErrorCode::Success),
			Err(e) => {
				if self.ext.debug_buffer_enabled() {
					self.ext.append_debug_buffer("call failed with: ");
					self.ext.append_debug_buffer(e.into());
				};
				Ok(ErrorReturnCode::get())
			},
		}
	}

	/// Write the given buffer and its length to the designated locations in sandbox memory and
	/// charge gas according to the token returned by `create_token`.
	//
	/// `out_ptr` is the location in sandbox memory where `buf` should be written to.
	/// `out_len_ptr` is an in-out location in sandbox memory. It is read to determine the
	/// length of the buffer located at `out_ptr`. If that buffer is large enough the actual
	/// `buf.len()` is written to this location.
	///
	/// If `out_ptr` is set to the sentinel value of `SENTINEL` and `allow_skip` is true the
	/// operation is skipped and `Ok` is returned. This is supposed to help callers to make copying
	/// output optional. For example to skip copying back the output buffer of an `seal_call`
	/// when the caller is not interested in the result.
	///
	/// `create_token` can optionally instruct this function to charge the gas meter with the token
	/// it returns. `create_token` receives the variable amount of bytes that are about to be copied
	/// by this function.
	///
	/// In addition to the error conditions of `Memory::write` this functions returns
	/// `Err` if the size of the buffer located at `out_ptr` is too small to fit `buf`.
	pub fn write_sandbox_output<M: Memory<E::T>>(
		&mut self,
		memory: &mut M,
		out_ptr: u32,
		out_len_ptr: u32,
		buf: &[u8],
		allow_skip: bool,
		create_token: impl FnOnce(u32) -> Option<RuntimeCosts>,
	) -> Result<(), DispatchError> {
		if allow_skip && out_ptr == SENTINEL {
			return Ok(())
		}

		let buf_len = buf.len() as u32;
		let len: u32 = memory.read_as(out_len_ptr)?;

		if len < buf_len {
			return Err(Error::<E::T>::OutputBufferTooSmall.into())
		}

		if let Some(costs) = create_token(buf_len) {
			self.charge_gas(costs)?;
		}

		memory.write(out_ptr, buf)?;
		memory.write(out_len_ptr, &buf_len.encode())
	}

	/// Computes the given hash function on the supplied input.
	///
	/// Reads from the sandboxed input buffer into an intermediate buffer.
	/// Returns the result directly to the output buffer of the sandboxed memory.
	///
	/// It is the callers responsibility to provide an output buffer that
	/// is large enough to hold the expected amount of bytes returned by the
	/// chosen hash function.
	///
	/// # Note
	///
	/// The `input` and `output` buffers may overlap.
	fn compute_hash_on_intermediate_buffer<F, R, M: Memory<E::T>>(
		&self,
		memory: &mut M,
		hash_fn: F,
		input_ptr: u32,
		input_len: u32,
		output_ptr: u32,
	) -> Result<(), DispatchError>
	where
		F: FnOnce(&[u8]) -> R,
		R: AsRef<[u8]>,
	{
		// Copy input into supervisor memory.
		let input = memory.read(input_ptr, input_len)?;
		// Compute the hash on the input buffer using the given hash function.
		let hash = hash_fn(&input);
		// Write the resulting hash back into the sandboxed output buffer.
		memory.write(output_ptr, hash.as_ref())?;
		Ok(())
	}

	/// Fallible conversion of `DispatchError` to `ReturnErrorCode`.
	fn err_into_return_code(from: DispatchError) -> Result<ReturnErrorCode, DispatchError> {
		use ReturnErrorCode::*;

		let transfer_failed = Error::<E::T>::TransferFailed.into();
		let no_code = Error::<E::T>::CodeNotFound.into();
		let not_found = Error::<E::T>::ContractNotFound.into();

		match from {
			x if x == transfer_failed => Ok(TransferFailed),
			x if x == no_code => Ok(CodeNotFound),
			x if x == not_found => Ok(NotCallable),
			err => Err(err),
		}
	}

	/// Fallible conversion of a `ExecResult` to `ReturnErrorCode`.
	fn exec_into_return_code(from: ExecResult) -> Result<ReturnErrorCode, DispatchError> {
		use crate::exec::ErrorOrigin::Callee;

		let ExecError { error, origin } = match from {
			Ok(retval) => return Ok(retval.into()),
			Err(err) => err,
		};

		match (error, origin) {
			(_, Callee) => Ok(ReturnErrorCode::CalleeTrapped),
			(err, _) => Self::err_into_return_code(err),
		}
	}
	fn decode_key<M: Memory<E::T>>(
		&self,
		memory: &M,
		key_type: KeyType,
		key_ptr: u32,
	) -> Result<crate::exec::Key<E::T>, TrapReason> {
		let res = match key_type {
			KeyType::Fix => {
				let key = memory.read(key_ptr, 32u32)?;
				Key::try_from_fix(key)
			},
			KeyType::Var(len) => {
				ensure!(
					len <= <<E as Ext>::T as Config>::MaxStorageKeyLen::get(),
					Error::<E::T>::DecodingFailed
				);
				let key = memory.read(key_ptr, len)?;
				Key::try_from_var(key)
			},
		};

		res.map_err(|_| Error::<E::T>::DecodingFailed.into())
	}

	fn set_storage<M: Memory<E::T>>(
		&mut self,
		memory: &M,
		key_type: KeyType,
		key_ptr: u32,
		value_ptr: u32,
		value_len: u32,
	) -> Result<u32, TrapReason> {
		let max_size = self.ext.max_value_size();
		let charged = self
			.charge_gas(RuntimeCosts::SetStorage { new_bytes: value_len, old_bytes: max_size })?;
		if value_len > max_size {
			return Err(Error::<E::T>::ValueTooLarge.into())
		}
		let key = self.decode_key(memory, key_type, key_ptr)?;
		let value = Some(memory.read(value_ptr, value_len)?);
		let write_outcome = self.ext.set_storage(&key, value, false)?;

		self.adjust_gas(
			charged,
			RuntimeCosts::SetStorage { new_bytes: value_len, old_bytes: write_outcome.old_len() },
		);
		Ok(write_outcome.old_len_with_sentinel())
	}

	fn clear_storage<M: Memory<E::T>>(
		&mut self,
		memory: &M,
		key_type: KeyType,
		key_ptr: u32,
	) -> Result<u32, TrapReason> {
		let charged = self.charge_gas(RuntimeCosts::ClearStorage(self.ext.max_value_size()))?;
		let key = self.decode_key(memory, key_type, key_ptr)?;
		let outcome = self.ext.set_storage(&key, None, false)?;

		self.adjust_gas(charged, RuntimeCosts::ClearStorage(outcome.old_len()));
		Ok(outcome.old_len_with_sentinel())
	}

	fn get_storage<M: Memory<E::T>>(
		&mut self,
		memory: &mut M,
		key_type: KeyType,
		key_ptr: u32,
		out_ptr: u32,
		out_len_ptr: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		let charged = self.charge_gas(RuntimeCosts::GetStorage(self.ext.max_value_size()))?;
		let key = self.decode_key(memory, key_type, key_ptr)?;
		let outcome = self.ext.get_storage(&key);

		if let Some(value) = outcome {
			self.adjust_gas(charged, RuntimeCosts::GetStorage(value.len() as u32));
			self.write_sandbox_output(
				memory,
				out_ptr,
				out_len_ptr,
				&value,
				false,
				already_charged,
			)?;
			Ok(ReturnErrorCode::Success)
		} else {
			self.adjust_gas(charged, RuntimeCosts::GetStorage(0));
			Ok(ReturnErrorCode::KeyNotFound)
		}
	}

	fn contains_storage<M: Memory<E::T>>(
		&mut self,
		memory: &M,
		key_type: KeyType,
		key_ptr: u32,
	) -> Result<u32, TrapReason> {
		let charged = self.charge_gas(RuntimeCosts::ContainsStorage(self.ext.max_value_size()))?;
		let key = self.decode_key(memory, key_type, key_ptr)?;
		let outcome = self.ext.get_storage_size(&key);

		self.adjust_gas(charged, RuntimeCosts::ContainsStorage(outcome.unwrap_or(0)));
		Ok(outcome.unwrap_or(SENTINEL))
	}

	fn set_transient_storage<M: Memory<E::T>>(
		&mut self,
		memory: &mut M,
		key_type: KeyType,
		key_ptr: u32,
		value_ptr: u32,
		value_len: u32,
	) -> Result<u32, TrapReason> {
		let max_size = self.ext.max_value_size();
		let charged = self.charge_gas(RuntimeCosts::SetTransientStorage {
			new_bytes: value_len,
			old_bytes: max_size,
		})?;
		if value_len > max_size {
			return Err(Error::<E::T>::ValueTooLarge.into())
		}
		let key = self.decode_key(memory, key_type, key_ptr)?;
		let value = Some(memory.read(value_ptr, value_len)?);
		let write_outcome = self.ext.set_transient_storage(&key, value, false)?;
		self.adjust_gas(
			charged,
			RuntimeCosts::SetTransientStorage {
				new_bytes: value_len,
				old_bytes: write_outcome.old_len(),
			},
		);
		Ok(write_outcome.old_len_with_sentinel())
	}

	fn clear_transient_storage<M: Memory<E::T>>(
		&mut self,
		memory: &mut M,
		key_type: KeyType,
		key_ptr: u32,
	) -> Result<u32, TrapReason> {
		let charged =
			self.charge_gas(RuntimeCosts::ClearTransientStorage(self.ext.max_value_size()))?;
		let key = self.decode_key(memory, key_type, key_ptr)?;
		let outcome = self.ext.set_transient_storage(&key, None, false)?;

		self.adjust_gas(charged, RuntimeCosts::ClearTransientStorage(outcome.old_len()));
		Ok(outcome.old_len_with_sentinel())
	}

	fn get_transient_storage<M: Memory<E::T>>(
		&mut self,
		memory: &mut M,
		key_type: KeyType,
		key_ptr: u32,
		out_ptr: u32,
		out_len_ptr: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		let charged =
			self.charge_gas(RuntimeCosts::GetTransientStorage(self.ext.max_value_size()))?;
		let key = self.decode_key(memory, key_type, key_ptr)?;
		let outcome = self.ext.get_transient_storage(&key);

		if let Some(value) = outcome {
			self.adjust_gas(charged, RuntimeCosts::GetTransientStorage(value.len() as u32));
			self.write_sandbox_output(
				memory,
				out_ptr,
				out_len_ptr,
				&value,
				false,
				already_charged,
			)?;
			Ok(ReturnErrorCode::Success)
		} else {
			self.adjust_gas(charged, RuntimeCosts::GetTransientStorage(0));
			Ok(ReturnErrorCode::KeyNotFound)
		}
	}

	fn contains_transient_storage<M: Memory<E::T>>(
		&mut self,
		memory: &mut M,
		key_type: KeyType,
		key_ptr: u32,
	) -> Result<u32, TrapReason> {
		let charged =
			self.charge_gas(RuntimeCosts::ContainsTransientStorage(self.ext.max_value_size()))?;
		let key = self.decode_key(memory, key_type, key_ptr)?;
		let outcome = self.ext.get_transient_storage_size(&key);

		self.adjust_gas(charged, RuntimeCosts::ContainsTransientStorage(outcome.unwrap_or(0)));
		Ok(outcome.unwrap_or(SENTINEL))
	}

	fn take_transient_storage<M: Memory<E::T>>(
		&mut self,
		memory: &mut M,
		key_type: KeyType,
		key_ptr: u32,
		out_ptr: u32,
		out_len_ptr: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		let charged =
			self.charge_gas(RuntimeCosts::TakeTransientStorage(self.ext.max_value_size()))?;
		let key = self.decode_key(memory, key_type, key_ptr)?;
		if let crate::storage::WriteOutcome::Taken(value) =
			self.ext.set_transient_storage(&key, None, true)?
		{
			self.adjust_gas(charged, RuntimeCosts::TakeTransientStorage(value.len() as u32));
			self.write_sandbox_output(
				memory,
				out_ptr,
				out_len_ptr,
				&value,
				false,
				already_charged,
			)?;
			Ok(ReturnErrorCode::Success)
		} else {
			self.adjust_gas(charged, RuntimeCosts::TakeTransientStorage(0));
			Ok(ReturnErrorCode::KeyNotFound)
		}
	}

	fn call<M: Memory<E::T>>(
		&mut self,
		memory: &mut M,
		flags: CallFlags,
		call_type: CallType,
		input_data_ptr: u32,
		input_data_len: u32,
		output_ptr: u32,
		output_len_ptr: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		self.charge_gas(call_type.cost())?;

		let input_data = if flags.contains(CallFlags::CLONE_INPUT) {
			let input = self.input_data.as_ref().ok_or(Error::<E::T>::InputForwarded)?;
			charge_gas!(self, RuntimeCosts::CallInputCloned(input.len() as u32))?;
			input.clone()
		} else if flags.contains(CallFlags::FORWARD_INPUT) {
			self.input_data.take().ok_or(Error::<E::T>::InputForwarded)?
		} else {
			self.charge_gas(RuntimeCosts::CopyFromContract(input_data_len))?;
			memory.read(input_data_ptr, input_data_len)?
		};

		let call_outcome = match call_type {
			CallType::Call { callee_ptr, value_ptr, deposit_ptr, weight } => {
				let callee: <<E as Ext>::T as frame_system::Config>::AccountId =
					memory.read_as(callee_ptr)?;
				let deposit_limit: BalanceOf<<E as Ext>::T> = if deposit_ptr == SENTINEL {
					BalanceOf::<<E as Ext>::T>::zero()
				} else {
					memory.read_as(deposit_ptr)?
				};
				let read_only = flags.contains(CallFlags::READ_ONLY);
				let value: BalanceOf<<E as Ext>::T> = memory.read_as(value_ptr)?;
				if value > 0u32.into() {
					// If the call value is non-zero and state change is not allowed, issue an
					// error.
					if read_only || self.ext.is_read_only() {
						return Err(Error::<E::T>::StateChangeDenied.into());
					}
					self.charge_gas(RuntimeCosts::CallTransferSurcharge)?;
				}
				self.ext.call(
					weight,
					deposit_limit,
					callee,
					value,
					input_data,
					flags.contains(CallFlags::ALLOW_REENTRY),
					read_only,
				)
			},
			CallType::DelegateCall { code_hash_ptr } => {
				if flags.intersects(CallFlags::ALLOW_REENTRY | CallFlags::READ_ONLY) {
					return Err(Error::<E::T>::InvalidCallFlags.into())
				}
				let code_hash = memory.read_as(code_hash_ptr)?;
				self.ext.delegate_call(code_hash, input_data)
			},
		};

		// `TAIL_CALL` only matters on an `OK` result. Otherwise the call stack comes to
		// a halt anyways without anymore code being executed.
		if flags.contains(CallFlags::TAIL_CALL) {
			if let Ok(return_value) = call_outcome {
				return Err(TrapReason::Return(ReturnData {
					flags: return_value.flags.bits(),
					data: return_value.data,
				}))
			}
		}

		if let Ok(output) = &call_outcome {
			self.write_sandbox_output(
				memory,
				output_ptr,
				output_len_ptr,
				&output.data,
				true,
				|len| Some(RuntimeCosts::CopyToContract(len)),
			)?;
		}
		Ok(Self::exec_into_return_code(call_outcome)?)
	}

	fn instantiate<M: Memory<E::T>>(
		&mut self,
		memory: &mut M,
		code_hash_ptr: u32,
		weight: Weight,
		deposit_ptr: u32,
		value_ptr: u32,
		input_data_ptr: u32,
		input_data_len: u32,
		address_ptr: u32,
		address_len_ptr: u32,
		output_ptr: u32,
		output_len_ptr: u32,
		salt_ptr: u32,
		salt_len: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		self.charge_gas(RuntimeCosts::Instantiate { input_data_len, salt_len })?;
		let deposit_limit: BalanceOf<<E as Ext>::T> = if deposit_ptr == SENTINEL {
			BalanceOf::<<E as Ext>::T>::zero()
		} else {
			memory.read_as(deposit_ptr)?
		};
		let value: BalanceOf<<E as Ext>::T> = memory.read_as(value_ptr)?;
		let code_hash: CodeHash<<E as Ext>::T> = memory.read_as(code_hash_ptr)?;
		let input_data = memory.read(input_data_ptr, input_data_len)?;
		let salt = memory.read(salt_ptr, salt_len)?;
		let instantiate_outcome =
			self.ext.instantiate(weight, deposit_limit, code_hash, value, input_data, &salt);
		if let Ok((address, output)) = &instantiate_outcome {
			if !output.flags.contains(ReturnFlags::REVERT) {
				self.write_sandbox_output(
					memory,
					address_ptr,
					address_len_ptr,
					&address.encode(),
					true,
					already_charged,
				)?;
			}
			self.write_sandbox_output(
				memory,
				output_ptr,
				output_len_ptr,
				&output.data,
				true,
				|len| Some(RuntimeCosts::CopyToContract(len)),
			)?;
		}
		Ok(Self::exec_into_return_code(instantiate_outcome.map(|(_, retval)| retval))?)
	}

	fn terminate<M: Memory<E::T>>(
		&mut self,
		memory: &M,
		beneficiary_ptr: u32,
	) -> Result<(), TrapReason> {
		let count = self.ext.locked_delegate_dependencies_count() as _;
		self.charge_gas(RuntimeCosts::Terminate(count))?;

		let beneficiary: <<E as Ext>::T as frame_system::Config>::AccountId =
			memory.read_as(beneficiary_ptr)?;
		self.ext.terminate(&beneficiary)?;
		Err(TrapReason::Termination)
	}
}

// This is the API exposed to contracts.
//
// # Note
//
// Any input that leads to a out of bound error (reading or writing) or failing to decode
// data passed to the supervisor will lead to a trap. This is not documented explicitly
// for every function.
#[define_env]
pub mod env {
	/// Noop function used to benchmark the time it takes to execute an empty function.
	#[cfg(feature = "runtime-benchmarks")]
	fn noop(&mut self, memory: &mut M) -> Result<(), TrapReason> {
		Ok(())
	}

	/// Set the value at the given key in the contract storage.
	/// See [`pallet_contracts_uapi::HostFn::set_storage_v2`]
	#[api_version(0)]
	#[mutating]
	fn set_storage(
		&mut self,
		memory: &mut M,
		key_ptr: u32,
		key_len: u32,
		value_ptr: u32,
		value_len: u32,
	) -> Result<u32, TrapReason> {
		if key_len == SENTINEL {
			self.set_storage(memory, KeyType::Fix, key_ptr, value_ptr, value_len)
		} else {
			self.set_storage(memory, KeyType::Var(key_len), key_ptr, value_ptr, value_len)
		}
	}

	/// Clear the value at the given key in the contract storage.
	/// See [`pallet_contracts_uapi::HostFn::clear_storage_v1`]
	#[api_version(0)]
	#[mutating]
	fn clear_storage(
		&mut self,
		memory: &mut M,
		key_ptr: u32,
		key_len: u32,
	) -> Result<u32, TrapReason> {
		if key_len == SENTINEL {
			self.clear_storage(memory, KeyType::Fix, key_ptr)
		} else {
			self.clear_storage(memory, KeyType::Var(key_len), key_ptr)
		}
	}

	/// Retrieve the value under the given key from storage.
	/// See [`pallet_contracts_uapi::HostFn::get_storage_v1`]
	#[api_version(0)]
	fn get_storage(
		&mut self,
		memory: &mut M,
		key_ptr: u32,
		key_len: u32,
		out_ptr: u32,
		out_len_ptr: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		if key_len == SENTINEL {
			self.get_storage(memory, KeyType::Fix, key_ptr, out_ptr, out_len_ptr)
		} else {
			self.get_storage(memory, KeyType::Var(key_len), key_ptr, out_ptr, out_len_ptr)
		}
	}

	/// Checks whether there is a value stored under the given key.
	/// See [`pallet_contracts_uapi::HostFn::contains_storage_v1`]
	#[api_version(0)]
	fn contains_storage(
		&mut self,
		memory: &mut M,
		key_ptr: u32,
		key_len: u32,
	) -> Result<u32, TrapReason> {
		if key_len == SENTINEL {
			self.contains_storage(memory, KeyType::Fix, key_ptr)
		} else {
			self.contains_storage(memory, KeyType::Var(key_len), key_ptr)
		}
	}

	/// Retrieve and remove the value under the given key from storage.
	/// See [`pallet_contracts_uapi::HostFn::take_storage`]
	#[api_version(0)]
	#[mutating]
	fn take_storage(
		&mut self,
		memory: &mut M,
		key_ptr: u32,
		key_len: u32,
		out_ptr: u32,
		out_len_ptr: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		let charged = self.charge_gas(RuntimeCosts::TakeStorage(self.ext.max_value_size()))?;
		ensure!(
			key_len <= <<E as Ext>::T as Config>::MaxStorageKeyLen::get(),
			Error::<E::T>::DecodingFailed
		);
		let key = memory.read(key_ptr, key_len)?;
		if let crate::storage::WriteOutcome::Taken(value) = self.ext.set_storage(
			&Key::<E::T>::try_from_var(key).map_err(|_| Error::<E::T>::DecodingFailed)?,
			None,
			true,
		)? {
			self.adjust_gas(charged, RuntimeCosts::TakeStorage(value.len() as u32));
			self.write_sandbox_output(
				memory,
				out_ptr,
				out_len_ptr,
				&value,
				false,
				already_charged,
			)?;
			Ok(ReturnErrorCode::Success)
		} else {
			self.adjust_gas(charged, RuntimeCosts::TakeStorage(0));
			Ok(ReturnErrorCode::KeyNotFound)
		}
	}

	/// Set the value at the given key in the contract transient storage.
	fn set_transient_storage(
		&mut self,
		memory: &mut M,
		key_ptr: u32,
		key_len: u32,
		value_ptr: u32,
		value_len: u32,
	) -> Result<u32, TrapReason> {
		self.set_transient_storage(memory, KeyType::Var(key_len), key_ptr, value_ptr, value_len)
	}

	/// Clear the value at the given key in the contract storage.
	fn clear_transient_storage(
		&mut self,
		memory: &mut M,
		key_ptr: u32,
		key_len: u32,
	) -> Result<u32, TrapReason> {
		self.clear_transient_storage(memory, KeyType::Var(key_len), key_ptr)
	}

	/// Retrieve the value under the given key from transient storage.
	fn get_transient_storage(
		&mut self,
		memory: &mut M,
		key_ptr: u32,
		key_len: u32,
		out_ptr: u32,
		out_len_ptr: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		self.get_transient_storage(memory, KeyType::Var(key_len), key_ptr, out_ptr, out_len_ptr)
	}

	/// Checks whether there is a value stored under the given key in transient storage.
	fn contains_transient_storage(
		&mut self,
		memory: &mut M,
		key_ptr: u32,
		key_len: u32,
	) -> Result<u32, TrapReason> {
		self.contains_transient_storage(memory, KeyType::Var(key_len), key_ptr)
	}

	/// Retrieve and remove the value under the given key from transient storage.
	fn take_transient_storage(
		&mut self,
		memory: &mut M,
		key_ptr: u32,
		key_len: u32,
		out_ptr: u32,
		out_len_ptr: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		self.take_transient_storage(memory, KeyType::Var(key_len), key_ptr, out_ptr, out_len_ptr)
	}

	/// Transfer some value to another account.
	/// See [`pallet_contracts_uapi::HostFn::transfer`].
	#[api_version(0)]
	#[mutating]
	fn transfer(
		&mut self,
		memory: &mut M,
		account_ptr: u32,
		value_ptr: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		self.charge_gas(RuntimeCosts::Transfer)?;
		let callee: <<E as Ext>::T as frame_system::Config>::AccountId =
			memory.read_as(account_ptr)?;
		let value: BalanceOf<<E as Ext>::T> = memory.read_as(value_ptr)?;
		let result = self.ext.transfer(&callee, value);
		match result {
			Ok(()) => Ok(ReturnErrorCode::Success),
			Err(err) => {
				let code = Runtime::<E>::err_into_return_code(err)?;
				Ok(code)
			},
		}
	}

	/// Make a call to another contract.
	/// See [`pallet_contracts_uapi::HostFn::call_v2`].
	#[api_version(0)]
	fn call(
		&mut self,
		memory: &mut M,
		flags: u32,
		callee_ptr: u32,
		ref_time_limit: u64,
		proof_size_limit: u64,
		deposit_ptr: u32,
		value_ptr: u32,
		input_data_ptr: u32,
		input_data_len: u32,
		output_ptr: u32,
		output_len_ptr: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		self.call(
			memory,
			CallFlags::from_bits(flags).ok_or(Error::<E::T>::InvalidCallFlags)?,
			CallType::Call {
				callee_ptr,
				value_ptr,
				deposit_ptr,
				weight: Weight::from_parts(ref_time_limit, proof_size_limit),
			},
			input_data_ptr,
			input_data_len,
			output_ptr,
			output_len_ptr,
		)
	}

	/// Execute code in the context (storage, caller, value) of the current contract.
	/// See [`pallet_contracts_uapi::HostFn::delegate_call`].
	#[api_version(0)]
	fn delegate_call(
		&mut self,
		memory: &mut M,
		flags: u32,
		code_hash_ptr: u32,
		input_data_ptr: u32,
		input_data_len: u32,
		output_ptr: u32,
		output_len_ptr: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		self.call(
			memory,
			CallFlags::from_bits(flags).ok_or(Error::<E::T>::InvalidCallFlags)?,
			CallType::DelegateCall { code_hash_ptr },
			input_data_ptr,
			input_data_len,
			output_ptr,
			output_len_ptr,
		)
	}

	/// Instantiate a contract with the specified code hash.
	/// See [`pallet_contracts_uapi::HostFn::instantiate_v2`].
	#[api_version(0)]
	#[mutating]
	fn instantiate(
		&mut self,
		memory: &mut M,
		code_hash_ptr: u32,
		ref_time_limit: u64,
		proof_size_limit: u64,
		deposit_ptr: u32,
		value_ptr: u32,
		input_data_ptr: u32,
		input_data_len: u32,
		address_ptr: u32,
		address_len_ptr: u32,
		output_ptr: u32,
		output_len_ptr: u32,
		salt_ptr: u32,
		salt_len: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		self.instantiate(
			memory,
			code_hash_ptr,
			Weight::from_parts(ref_time_limit, proof_size_limit),
			deposit_ptr,
			value_ptr,
			input_data_ptr,
			input_data_len,
			address_ptr,
			address_len_ptr,
			output_ptr,
			output_len_ptr,
			salt_ptr,
			salt_len,
		)
	}

	/// Remove the calling account and transfer remaining **free** balance.
	/// See [`pallet_contracts_uapi::HostFn::terminate_v1`].
	#[api_version(0)]
	#[mutating]
	fn terminate(&mut self, memory: &mut M, beneficiary_ptr: u32) -> Result<(), TrapReason> {
		self.terminate(memory, beneficiary_ptr)
	}

	/// Stores the input passed by the caller into the supplied buffer.
	/// See [`pallet_contracts_uapi::HostFn::input`].
	#[api_version(0)]
	fn input(&mut self, memory: &mut M, out_ptr: u32, out_len_ptr: u32) -> Result<(), TrapReason> {
		if let Some(input) = self.input_data.take() {
			self.write_sandbox_output(memory, out_ptr, out_len_ptr, &input, false, |len| {
				Some(RuntimeCosts::CopyToContract(len))
			})?;
			self.input_data = Some(input);
			Ok(())
		} else {
			Err(Error::<E::T>::InputForwarded.into())
		}
	}

	/// Cease contract execution and save a data buffer as a result of the execution.
	/// See [`pallet_contracts_uapi::HostFn::return_value`].
	#[api_version(0)]
	fn seal_return(
		&mut self,
		memory: &mut M,
		flags: u32,
		data_ptr: u32,
		data_len: u32,
	) -> Result<(), TrapReason> {
		self.charge_gas(RuntimeCosts::CopyFromContract(data_len))?;
		Err(TrapReason::Return(ReturnData { flags, data: memory.read(data_ptr, data_len)? }))
	}

	/// Stores the address of the caller into the supplied buffer.
	/// See [`pallet_contracts_uapi::HostFn::caller`].
	#[api_version(0)]
	fn caller(&mut self, memory: &mut M, out_ptr: u32, out_len_ptr: u32) -> Result<(), TrapReason> {
		self.charge_gas(RuntimeCosts::Caller)?;
		let caller = self.ext.caller().account_id()?.clone();
		Ok(self.write_sandbox_output(
			memory,
			out_ptr,
			out_len_ptr,
			&caller.encode(),
			false,
			already_charged,
		)?)
	}

	/// Checks whether a specified address belongs to a contract.
	/// See [`pallet_contracts_uapi::HostFn::is_contract`].
	#[api_version(0)]
	fn is_contract(&mut self, memory: &mut M, account_ptr: u32) -> Result<u32, TrapReason> {
		self.charge_gas(RuntimeCosts::IsContract)?;
		let address: <<E as Ext>::T as frame_system::Config>::AccountId =
			memory.read_as(account_ptr)?;

		Ok(self.ext.is_contract(&address) as u32)
	}

	/// Retrieve the code hash for a specified contract address.
	/// See [`pallet_contracts_uapi::HostFn::code_hash`].
	#[api_version(0)]
	fn code_hash(
		&mut self,
		memory: &mut M,
		account_ptr: u32,
		out_ptr: u32,
		out_len_ptr: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		self.charge_gas(RuntimeCosts::CodeHash)?;
		let address: <<E as Ext>::T as frame_system::Config>::AccountId =
			memory.read_as(account_ptr)?;
		if let Some(value) = self.ext.code_hash(&address) {
			self.write_sandbox_output(
				memory,
				out_ptr,
				out_len_ptr,
				&value.encode(),
				false,
				already_charged,
			)?;
			Ok(ReturnErrorCode::Success)
		} else {
			Ok(ReturnErrorCode::KeyNotFound)
		}
	}

	/// Retrieve the code hash of the currently executing contract.
	/// See [`pallet_contracts_uapi::HostFn::own_code_hash`].
	#[api_version(0)]
	fn own_code_hash(
		&mut self,
		memory: &mut M,
		out_ptr: u32,
		out_len_ptr: u32,
	) -> Result<(), TrapReason> {
		self.charge_gas(RuntimeCosts::OwnCodeHash)?;
		let code_hash_encoded = &self.ext.own_code_hash().encode();
		Ok(self.write_sandbox_output(
			memory,
			out_ptr,
			out_len_ptr,
			code_hash_encoded,
			false,
			already_charged,
		)?)
	}

	/// Checks whether the caller of the current contract is the origin of the whole call stack.
	/// See [`pallet_contracts_uapi::HostFn::caller_is_origin`].
	#[api_version(0)]
	fn caller_is_origin(&mut self, _memory: &mut M) -> Result<u32, TrapReason> {
		self.charge_gas(RuntimeCosts::CallerIsOrigin)?;
		Ok(self.ext.caller_is_origin() as u32)
	}

	/// Checks whether the caller of the current contract is root.
	/// See [`pallet_contracts_uapi::HostFn::caller_is_root`].
	#[api_version(0)]
	fn caller_is_root(&mut self, _memory: &mut M) -> Result<u32, TrapReason> {
		self.charge_gas(RuntimeCosts::CallerIsRoot)?;
		Ok(self.ext.caller_is_root() as u32)
	}

	/// Stores the address of the current contract into the supplied buffer.
	/// See [`pallet_contracts_uapi::HostFn::address`].
	#[api_version(0)]
	fn address(
		&mut self,
		memory: &mut M,
		out_ptr: u32,
		out_len_ptr: u32,
	) -> Result<(), TrapReason> {
		self.charge_gas(RuntimeCosts::Address)?;
		Ok(self.write_sandbox_output(
			memory,
			out_ptr,
			out_len_ptr,
			&self.ext.address().encode(),
			false,
			already_charged,
		)?)
	}

	/// Stores the price for the specified amount of gas into the supplied buffer.
	/// See [`pallet_contracts_uapi::HostFn::weight_to_fee`].
	#[api_version(0)]
	fn weight_to_fee_reftime(
		&mut self,
		memory: &mut M,
		gas: u64,
		out_ptr: u32,
		out_len_ptr: u32,
	) -> Result<(), TrapReason> {
		let gas = Weight::from_parts(gas, 0);
		self.charge_gas(RuntimeCosts::WeightToFee)?;
		Ok(self.write_sandbox_output(
			memory,
			out_ptr,
			out_len_ptr,
			&self.ext.get_weight_price(gas).encode(),
			false,
			already_charged,
		)?)
	}

	/// Stores the price for the specified amount of weight into the supplied buffer.
	/// See [`pallet_contracts_uapi::HostFn::weight_to_fee_v1`].
	fn weight_to_fee_full(
		&mut self,
		memory: &mut M,
		ref_time_limit: u64,
		proof_size_limit: u64,
		out_ptr: u32,
		out_len_ptr: u32,
	) -> Result<(), TrapReason> {
		let weight = Weight::from_parts(ref_time_limit, proof_size_limit);
		self.charge_gas(RuntimeCosts::WeightToFee)?;
		Ok(self.write_sandbox_output(
			memory,
			out_ptr,
			out_len_ptr,
			&self.ext.get_weight_price(weight).encode(),
			false,
			already_charged,
		)?)
	}

	/// Stores the weight left into the supplied buffer.
	/// See [`pallet_contracts_uapi::HostFn::gas_left`].
	#[api_version(0)]
	fn gas_left_reftime(
		&mut self,
		memory: &mut M,
		out_ptr: u32,
		out_len_ptr: u32,
	) -> Result<(), TrapReason> {
		self.charge_gas(RuntimeCosts::GasLeft)?;
		let gas_left = &self.ext.gas_meter().gas_left().ref_time().encode();
		Ok(self.write_sandbox_output(
			memory,
			out_ptr,
			out_len_ptr,
			gas_left,
			false,
			already_charged,
		)?)
	}

	/// Stores the amount of weight left into the supplied buffer.
	/// See [`pallet_contracts_uapi::HostFn::gas_left_v1`].
	fn gas_left_full(
		&mut self,
		memory: &mut M,
		out_ptr: u32,
		out_len_ptr: u32,
	) -> Result<(), TrapReason> {
		self.charge_gas(RuntimeCosts::GasLeft)?;
		let gas_left = &self.ext.gas_meter().gas_left().encode();
		Ok(self.write_sandbox_output(
			memory,
			out_ptr,
			out_len_ptr,
			gas_left,
			false,
			already_charged,
		)?)
	}

	/// Stores the *free* balance of the current account into the supplied buffer.
	/// See [`pallet_contracts_uapi::HostFn::balance`].
	#[api_version(0)]
	fn balance(
		&mut self,
		memory: &mut M,
		out_ptr: u32,
		out_len_ptr: u32,
	) -> Result<(), TrapReason> {
		self.charge_gas(RuntimeCosts::Balance)?;
		Ok(self.write_sandbox_output(
			memory,
			out_ptr,
			out_len_ptr,
			&self.ext.balance().encode(),
			false,
			already_charged,
		)?)
	}

	/// Stores the value transferred along with this call/instantiate into the supplied buffer.
	/// See [`pallet_contracts_uapi::HostFn::value_transferred`].
	#[api_version(0)]
	fn value_transferred(
		&mut self,
		memory: &mut M,
		out_ptr: u32,
		out_len_ptr: u32,
	) -> Result<(), TrapReason> {
		self.charge_gas(RuntimeCosts::ValueTransferred)?;
		Ok(self.write_sandbox_output(
			memory,
			out_ptr,
			out_len_ptr,
			&self.ext.value_transferred().encode(),
			false,
			already_charged,
		)?)
	}

	/// Load the latest block timestamp into the supplied buffer
	/// See [`pallet_contracts_uapi::HostFn::now`].
	#[api_version(0)]
	fn now(&mut self, memory: &mut M, out_ptr: u32, out_len_ptr: u32) -> Result<(), TrapReason> {
		self.charge_gas(RuntimeCosts::Now)?;
		Ok(self.write_sandbox_output(
			memory,
			out_ptr,
			out_len_ptr,
			&self.ext.now().encode(),
			false,
			already_charged,
		)?)
	}

	/// Stores the minimum balance (a.k.a. existential deposit) into the supplied buffer.
	/// See [`pallet_contracts_uapi::HostFn::minimum_balance`].
	#[api_version(0)]
	fn minimum_balance(
		&mut self,
		memory: &mut M,
		out_ptr: u32,
		out_len_ptr: u32,
	) -> Result<(), TrapReason> {
		self.charge_gas(RuntimeCosts::MinimumBalance)?;
		Ok(self.write_sandbox_output(
			memory,
			out_ptr,
			out_len_ptr,
			&self.ext.minimum_balance().encode(),
			false,
			already_charged,
		)?)
	}

	/// Deposit a contract event with the data buffer and optional list of topics.
	/// See [pallet_contracts_uapi::HostFn::deposit_event]
	#[api_version(0)]
	#[mutating]
	fn deposit_event(
		&mut self,
		memory: &mut M,
		topics_ptr: u32,
		topics_len: u32,
		data_ptr: u32,
		data_len: u32,
	) -> Result<(), TrapReason> {
		let num_topic = topics_len
			.checked_div(core::mem::size_of::<TopicOf<E::T>>() as u32)
			.ok_or("Zero sized topics are not allowed")?;
		self.charge_gas(RuntimeCosts::DepositEvent { num_topic, len: data_len })?;
		if data_len > self.ext.max_value_size() {
			return Err(Error::<E::T>::ValueTooLarge.into());
		}

		let topics: Vec<TopicOf<<E as Ext>::T>> = match topics_len {
			0 => Vec::new(),
			_ => memory.read_as_unbounded(topics_ptr, topics_len)?,
		};

		// If there are more than `event_topics`, then trap.
		if topics.len() > self.ext.schedule().limits.event_topics as usize {
			return Err(Error::<E::T>::TooManyTopics.into());
		}

		let event_data = memory.read(data_ptr, data_len)?;

		self.ext.deposit_event(topics, event_data);

		Ok(())
	}

	/// Stores the current block number of the current contract into the supplied buffer.
	/// See [`pallet_contracts_uapi::HostFn::block_number`].
	#[api_version(0)]
	fn block_number(
		&mut self,
		memory: &mut M,
		out_ptr: u32,
		out_len_ptr: u32,
	) -> Result<(), TrapReason> {
		self.charge_gas(RuntimeCosts::BlockNumber)?;
		Ok(self.write_sandbox_output(
			memory,
			out_ptr,
			out_len_ptr,
			&self.ext.block_number().encode(),
			false,
			already_charged,
		)?)
	}

	/// Computes the SHA2 256-bit hash on the given input buffer.
	/// See [`pallet_contracts_uapi::HostFn::hash_sha2_256`].
	#[api_version(0)]
	fn hash_sha2_256(
		&mut self,
		memory: &mut M,
		input_ptr: u32,
		input_len: u32,
		output_ptr: u32,
	) -> Result<(), TrapReason> {
		self.charge_gas(RuntimeCosts::HashSha256(input_len))?;
		Ok(self.compute_hash_on_intermediate_buffer(
			memory, sha2_256, input_ptr, input_len, output_ptr,
		)?)
	}

	/// Computes the KECCAK 256-bit hash on the given input buffer.
	/// See [`pallet_contracts_uapi::HostFn::hash_keccak_256`].
	#[api_version(0)]
	fn hash_keccak_256(
		&mut self,
		memory: &mut M,
		input_ptr: u32,
		input_len: u32,
		output_ptr: u32,
	) -> Result<(), TrapReason> {
		self.charge_gas(RuntimeCosts::HashKeccak256(input_len))?;
		Ok(self.compute_hash_on_intermediate_buffer(
			memory, keccak_256, input_ptr, input_len, output_ptr,
		)?)
	}

	/// Computes the BLAKE2 256-bit hash on the given input buffer.
	/// See [`pallet_contracts_uapi::HostFn::hash_blake2_256`].
	#[api_version(0)]
	fn hash_blake2_256(
		&mut self,
		memory: &mut M,
		input_ptr: u32,
		input_len: u32,
		output_ptr: u32,
	) -> Result<(), TrapReason> {
		self.charge_gas(RuntimeCosts::HashBlake256(input_len))?;
		Ok(self.compute_hash_on_intermediate_buffer(
			memory, blake2_256, input_ptr, input_len, output_ptr,
		)?)
	}

	/// Computes the BLAKE2 128-bit hash on the given input buffer.
	/// See [`pallet_contracts_uapi::HostFn::hash_blake2_128`].
	#[api_version(0)]
	fn hash_blake2_128(
		&mut self,
		memory: &mut M,
		input_ptr: u32,
		input_len: u32,
		output_ptr: u32,
	) -> Result<(), TrapReason> {
		self.charge_gas(RuntimeCosts::HashBlake128(input_len))?;
		Ok(self.compute_hash_on_intermediate_buffer(
			memory, blake2_128, input_ptr, input_len, output_ptr,
		)?)
	}

	/// Call into the chain extension provided by the chain if any.
	/// See [`pallet_contracts_uapi::HostFn::call_chain_extension`].
	fn call_chain_extension(
		&mut self,
		memory: &mut M,
		id: u32,
		input_ptr: u32,
		input_len: u32,
		output_ptr: u32,
		output_len_ptr: u32,
	) -> Result<u32, TrapReason> {
		use crate::chain_extension::{ChainExtension, Environment, RetVal};
		if !<E::T as Config>::ChainExtension::enabled() {
			return Err(Error::<E::T>::NoChainExtension.into());
		}
		let mut chain_extension = self.chain_extension.take().expect(
            "Constructor initializes with `Some`. This is the only place where it is set to `None`.\
			It is always reset to `Some` afterwards. qed",
        );
		let env =
			Environment::new(self, memory, id, input_ptr, input_len, output_ptr, output_len_ptr);
		let ret = match chain_extension.call(env)? {
			RetVal::Converging(val) => Ok(val),
			RetVal::Diverging { flags, data } =>
				Err(TrapReason::Return(ReturnData { flags: flags.bits(), data })),
		};
		self.chain_extension = Some(chain_extension);
		ret
	}

	/// Emit a custom debug message.
	///
	/// No newlines are added to the supplied message.
	/// Specifying invalid UTF-8 just drops the message with no trap.
	///
	/// This is a no-op if debug message recording is disabled which is always the case
	/// when the code is executing on-chain. The message is interpreted as UTF-8 and
	/// appended to the debug buffer which is then supplied to the calling RPC client.
	///
	/// # Note
	///
	/// Even though no action is taken when debug message recording is disabled there is still
	/// a non trivial overhead (and weight cost) associated with calling this function. Contract
	/// languages should remove calls to this function (either at runtime or compile time) when
	/// not being executed as an RPC. For example, they could allow users to disable logging
	/// through compile time flags (cargo features) for on-chain deployment. Additionally, the
	/// return value of this function can be cached in order to prevent further calls at runtime.
	#[api_version(0)]
	fn debug_message(
		&mut self,
		memory: &mut M,
		str_ptr: u32,
		str_len: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		let str_len = str_len.min(DebugBufferVec::<E::T>::bound() as u32);
		self.charge_gas(RuntimeCosts::DebugMessage(str_len))?;
		if self.ext.append_debug_buffer("") {
			let data = memory.read(str_ptr, str_len)?;
			if let Some(msg) = core::str::from_utf8(&data).ok() {
				self.ext.append_debug_buffer(msg);
			}
		}
		Ok(ReturnErrorCode::Success)
	}

	/// Call some dispatchable of the runtime.
	/// See [`frame_support::traits::call_runtime`].
	#[mutating]
	fn call_runtime(
		&mut self,
		memory: &mut M,
		call_ptr: u32,
		call_len: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		use frame_support::dispatch::GetDispatchInfo;
		self.charge_gas(RuntimeCosts::CopyFromContract(call_len))?;
		let call: <E::T as Config>::RuntimeCall = memory.read_as_unbounded(call_ptr, call_len)?;
		self.call_dispatchable::<CallRuntimeFailed>(
			call.get_dispatch_info(),
			RuntimeCosts::CallRuntime,
			|runtime| runtime.ext.call_runtime(call),
		)
	}

	/// Execute an XCM program locally, using the contract's address as the origin.
	/// See [`pallet_contracts_uapi::HostFn::execute_xcm`].
	#[mutating]
	fn xcm_execute(
		&mut self,
		memory: &mut M,
		msg_ptr: u32,
		msg_len: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		use frame_support::dispatch::DispatchInfo;
		use xcm::VersionedXcm;
		use xcm_builder::{ExecuteController, ExecuteControllerWeightInfo};

		self.charge_gas(RuntimeCosts::CopyFromContract(msg_len))?;
		let message: VersionedXcm<CallOf<E::T>> = memory.read_as_unbounded(msg_ptr, msg_len)?;

		let execute_weight =
			<<E::T as Config>::Xcm as ExecuteController<_, _>>::WeightInfo::execute();
		let weight = self.ext.gas_meter().gas_left().max(execute_weight);
		let dispatch_info = DispatchInfo { weight, ..Default::default() };

		self.call_dispatchable::<XcmExecutionFailed>(
			dispatch_info,
			RuntimeCosts::CallXcmExecute,
			|runtime| {
				let origin = crate::RawOrigin::Signed(runtime.ext.address().clone()).into();
				let weight_used = <<E::T as Config>::Xcm>::execute(
					origin,
					Box::new(message),
					weight.saturating_sub(execute_weight),
				)?;

				Ok(Some(weight_used.saturating_add(execute_weight)).into())
			},
		)
	}

	/// Send an XCM program from the contract to the specified destination.
	/// See [`pallet_contracts_uapi::HostFn::send_xcm`].
	#[mutating]
	fn xcm_send(
		&mut self,
		memory: &mut M,
		dest_ptr: u32,
		msg_ptr: u32,
		msg_len: u32,
		output_ptr: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		use xcm::{VersionedLocation, VersionedXcm};
		use xcm_builder::{SendController, SendControllerWeightInfo};

		self.charge_gas(RuntimeCosts::CopyFromContract(msg_len))?;
		let dest: VersionedLocation = memory.read_as(dest_ptr)?;

		let message: VersionedXcm<()> = memory.read_as_unbounded(msg_ptr, msg_len)?;
		let weight = <<E::T as Config>::Xcm as SendController<_>>::WeightInfo::send();
		self.charge_gas(RuntimeCosts::CallRuntime(weight))?;
		let origin = crate::RawOrigin::Signed(self.ext.address().clone()).into();

		match <<E::T as Config>::Xcm>::send(origin, dest.into(), message.into()) {
			Ok(message_id) => {
				memory.write(output_ptr, &message_id.encode())?;
				Ok(ReturnErrorCode::Success)
			},
			Err(e) => {
				if self.ext.append_debug_buffer("") {
					self.ext.append_debug_buffer("seal0::xcm_send failed with: ");
					self.ext.append_debug_buffer(e.into());
				};
				Ok(ReturnErrorCode::XcmSendFailed)
			},
		}
	}

	/// Recovers the ECDSA public key from the given message hash and signature.
	/// See [`pallet_contracts_uapi::HostFn::ecdsa_recover`].
	#[api_version(0)]
	fn ecdsa_recover(
		&mut self,
		memory: &mut M,
		signature_ptr: u32,
		message_hash_ptr: u32,
		output_ptr: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		self.charge_gas(RuntimeCosts::EcdsaRecovery)?;

		let mut signature: [u8; 65] = [0; 65];
		memory.read_into_buf(signature_ptr, &mut signature)?;
		let mut message_hash: [u8; 32] = [0; 32];
		memory.read_into_buf(message_hash_ptr, &mut message_hash)?;

		let result = self.ext.ecdsa_recover(&signature, &message_hash);

		match result {
			Ok(pub_key) => {
				// Write the recovered compressed ecdsa public key back into the sandboxed output
				// buffer.
				memory.write(output_ptr, pub_key.as_ref())?;

				Ok(ReturnErrorCode::Success)
			},
			Err(_) => Ok(ReturnErrorCode::EcdsaRecoveryFailed),
		}
	}

	/// Verify a sr25519 signature
	/// See [`pallet_contracts_uapi::HostFn::sr25519_verify`].
	#[api_version(0)]
	fn sr25519_verify(
		&mut self,
		memory: &mut M,
		signature_ptr: u32,
		pub_key_ptr: u32,
		message_len: u32,
		message_ptr: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		self.charge_gas(RuntimeCosts::Sr25519Verify(message_len))?;

		let mut signature: [u8; 64] = [0; 64];
		memory.read_into_buf(signature_ptr, &mut signature)?;

		let mut pub_key: [u8; 32] = [0; 32];
		memory.read_into_buf(pub_key_ptr, &mut pub_key)?;

		let message: Vec<u8> = memory.read(message_ptr, message_len)?;

		if self.ext.sr25519_verify(&signature, &message, &pub_key) {
			Ok(ReturnErrorCode::Success)
		} else {
			Ok(ReturnErrorCode::Sr25519VerifyFailed)
		}
	}

	/// Replace the contract code at the specified address with new code.
	/// See [`pallet_contracts_uapi::HostFn::set_code_hash`].
	#[api_version(0)]
	#[mutating]
	fn set_code_hash(
		&mut self,
		memory: &mut M,
		code_hash_ptr: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		self.charge_gas(RuntimeCosts::SetCodeHash)?;
		let code_hash: CodeHash<<E as Ext>::T> = memory.read_as(code_hash_ptr)?;
		match self.ext.set_code_hash(code_hash) {
			Err(err) => {
				let code = Runtime::<E>::err_into_return_code(err)?;
				Ok(code)
			},
			Ok(()) => Ok(ReturnErrorCode::Success),
		}
	}

	/// Calculates Ethereum address from the ECDSA compressed public key and stores
	/// See [`pallet_contracts_uapi::HostFn::ecdsa_to_eth_address`].
	#[api_version(0)]
	fn ecdsa_to_eth_address(
		&mut self,
		memory: &mut M,
		key_ptr: u32,
		out_ptr: u32,
	) -> Result<ReturnErrorCode, TrapReason> {
		self.charge_gas(RuntimeCosts::EcdsaToEthAddress)?;
		let mut compressed_key: [u8; 33] = [0; 33];
		memory.read_into_buf(key_ptr, &mut compressed_key)?;
		let result = self.ext.ecdsa_to_eth_address(&compressed_key);
		match result {
			Ok(eth_address) => {
				memory.write(out_ptr, eth_address.as_ref())?;
				Ok(ReturnErrorCode::Success)
			},
			Err(_) => Ok(ReturnErrorCode::EcdsaRecoveryFailed),
		}
	}

	/// Adds a new delegate dependency to the contract.
	/// See [`pallet_contracts_uapi::HostFn::lock_delegate_dependency`].
	#[api_version(0)]
	#[mutating]
	fn lock_delegate_dependency(
		&mut self,
		memory: &mut M,
		code_hash_ptr: u32,
	) -> Result<(), TrapReason> {
		self.charge_gas(RuntimeCosts::LockDelegateDependency)?;
		let code_hash = memory.read_as(code_hash_ptr)?;
		self.ext.lock_delegate_dependency(code_hash)?;
		Ok(())
	}

	/// Removes the delegate dependency from the contract.
	/// see [`pallet_contracts_uapi::HostFn::unlock_delegate_dependency`].
	#[api_version(0)]
	#[mutating]
	fn unlock_delegate_dependency(
		&mut self,
		memory: &mut M,
		code_hash_ptr: u32,
	) -> Result<(), TrapReason> {
		self.charge_gas(RuntimeCosts::UnlockDelegateDependency)?;
		let code_hash = memory.read_as(code_hash_ptr)?;
		self.ext.unlock_delegate_dependency(&code_hash)?;
		Ok(())
	}
}
